# RL_agentic_rag
This repository contains codebase for Bachelor Thesis on "Optimizing Agentic Retrieval-Augmented Generation using Reinforcement Learning-Based Methods". Research is conducted by Nataliia Bagan under supervision of Zhaochun Ren and Zhengliang Shi.

Currently 2 approaches are implemented. Their idea builds on RAG-Star, which adresses Multi-hop Question Answering. It first split user's query into a chain of subqueries. To find the best chain, it uses Monte Carlo Tree Search (MCTS). Approach 1 and 2 combine MCTS on subqueries with Search-R1 framework for search.

Currently used models:
*Retriever:* paraphrase-MiniLM-L3-v2
*Generation model:* SearchR1-nq_hotpotqa_train-llama3.2-3b-em-ppo
TODO: use ColBERT and Llama-3.1-8B

## Setup and execution

The environments are taken from [Search-R1 github](https://github.com/PeterGriffinJin/Search-R1):

### Retriever environment
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Dataset processing
*For now the implementation is under testing on a single multi-hop question, dataset instructions are to be added.*
TODO: use 

### Slurm job
Slurm job is provided as mcts_searchr1.slurm file with setup for GPU resource (some parameters are specific to ALICE computing facility, e.g. partition).

## Approach 1
### Structure of MCTS tree
Root node of the tree is the original user's query. Each further node represents subquery, which follows its parent's subquery in the chain. Thus, the full chain of subqueries representing the original query fully is a path from root node to a terminal node. 

For each node, search-r1 is called to find an answer to the node's subquery. Based on the relevance of subquery and given answer to the original (root) query, reward is assigned to the node. In the end, path with highest reward of the terminal node is chosen. The answer to this terminal node should be same as the answer to the original user's query, as it is the final subquery.

### Specific formulas/evaluations
Selection is based on UCB formula, where parameter C is currently chosen to be 2, weighted towards exploration.
TODO: tune parameter C.

Reward is evaluated by LLM, which is given original query, node's subquery and answer to it and asked to provide a relevance score between 0 and 1.

## Approach 2
Approach 2 uses the same MCTS tree structure and methods in the stages of MCTS. The difference is that instead of generating the answer for each of the node's subqueries, the answers are generated only in the end, for the nodes on the best path. The reward is the evaluated only based on relevance of a subquery to the original query (without an answer to it), which is given again by LLM, as in approach 1. This should be more efficient (as the time is saved on generation of each answer, which requires RAG), but may be less accurate, as less context for the reward evaluation is given.

## Approach 3
Approach 3 takes code from approach 2 as a base. The main distinction in the MCTS procedure is that multiple reward methods can be executed. Implemented rewards:
1. LLM-based reward (from approach 2) - reward_llm()
2. Exact string matching reward - reward_em()
3. Relaxed string matching reward - reward_em_relaxed()
4. Jaccard similarity (token overlap) - reward_jaccard()
5. Accuracy - reward_accuracy()
6. F1 score - reward_f1()

Additionally, the prompt was refined to better account for initially single-hop questions, which were earlier overcomplicated by the model.