import transformers
import torch
import random
import re
from datasets import load_dataset
import requests
import concurrent.futures
from ollama import chat
from ollama import ChatResponse

def initialize_model(model_id):
    """
    Initialize and return the tokenizer and model.
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
        model = model.to(device)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    return tokenizer, model, device

def run_ollama(prompt, model):
    """
    Generate answer to the prompt using ollama library.
    @param prompt: user's input
    """
    response = chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    response = response['message']['content']
    if model == 'qwen3:8b':
        response = response.split('</think>')[-1]
        response = '\n'.join([line for line in response.split('\n') if line.strip()])

    return response

def safe_run_ollama(prompt, model, timeout=120, retries=2):
    """
    Generate answer to the prompt using ollama with timeout and retry handling.
    @param prompt: User's input
    @param model: Model name (e.g., 'qwen3:8b')
    @param timeout: Max time (in seconds) to wait for a response
    @param retries: Number of retry attempts if timeout or error occurs
    """
    for attempt in range(1, retries + 1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_ollama, prompt, model)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"[Timeout] Attempt {attempt} exceeded {timeout} seconds.")
            except Exception as e:
                print(f"[Error] Attempt {attempt} failed with error: {e}")
    return f"[ERROR] All {retries} attempts failed or timed out."

def run_llm(prompt, tokenizer, model, device):
    """
    Generate answer to the prompt using the model (without stopping criteria).
    @param prompt: user's input
    """
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    # Encode the prompt and move it to the correct device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    
    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_text

def create_prompt(question):
    """
    Create the searchr1 prompt with the user's question.
    @param question: user's (sub)query
    """
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Prepare the message
    prompt = f"""Answer the given question. \
    You must conduct reasoning inside <think> and </think> first every time you get new information. \
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
    You can search as many times as your want. \
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    return prompt

class StopOnSequence(transformers.StoppingCriteria):
    """
    Define the custom stopping criterion
    """
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:7480/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

def clean_generated_output(output_text):
    """
    Clean up generated text that may have gone past stopping criteria
    and extract just the relevant answer.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1)

    return output_text

def run_search_llm(question, tokenizer, model, device):
    """
    Run the search LLM with the given question.
    """
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n{output_text}<information>{search_results}</information>\n'

    # Initialize the stopping criteria
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    prompt = create_prompt(question)

    cnt = 0

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('################# [Start Reasoning + Searching] ##################', flush=True)
    # print(prompt, flush=True)

    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(output_text, flush=True)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...', flush=True)
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        # print(search_text, flush=True)
    
    output_text = clean_generated_output(output_text)
    print(f'Search-R1 answer: {output_text}', flush=True)
    return output_text

def main():
    tokenizer, model, device = initialize_model("PeterJinGo/SearchR1-nq_hotpotqa_train-llama3.2-3b-em-ppo")

    # Define your question here
    question = "Which year was Steven Spielberg born?"
    # question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
    result = run_search_llm(question, tokenizer, model, device)
    # result = run_ollama(question, 'qwen3:8b')
    print(result, flush=True)

if __name__ == "__main__":
    main()

