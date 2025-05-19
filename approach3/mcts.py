import os
import random
import math
import json
import requests
import time
import pandas as pd
from graphviz import Digraph
from typing import List, Dict, Any, Optional, Tuple
import argparse
from infer import initialize_model, run_ollama, safe_run_ollama, run_llm, run_search_llm
from rewards import reward_function


class MCTSNode:
    """
    Node in the MCTS tree representing a subquery and its answer.
    @param query: subquery representing the current node
    @param answer: answer to the subquery of the current node

    @param parent: parent node in the MCTS tree
    @param children: list of child nodes in the MCTS tree
    @param prev_subqueries: list of subqueries leading to this node from the root

    @param visits: number of times this node has been visited
    @param reward: total reward received for this node
    """

    def __init__(self, query, parent=None, prev_subqueries=None):
        self.query = query
        self.answer = None

        self.parent = parent
        self.children = []
        self.prev_subqueries = prev_subqueries if prev_subqueries else []

        self.visits = 1 # start with 1 to avoid division by zero in UCB
        self.reward = 0.0

    def add_child(self, child_query, child_subqueries):
        child = MCTSNode(child_query, parent=self, prev_subqueries=child_subqueries)
        self.children.append(child)
        return child

    def update(self, reward, gamma=0.9):
        self.visits += 1
        self.reward = reward + gamma * self.reward

    def UCT(self, c_param=2):
        """
        Select the child node with the highest Upper Confidence bounds applied to Trees (UCT) value.
        """
        if not self.children:
            return None

        # UCB formula
        children_weights = [
            (child.reward / child.visits)
            + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[children_weights.index(max(children_weights))]

    def get_path(self):
        """
        Returns the path from root to this node as (subquery, answer) pairs.
        """
        if self.parent is None:
            return []

        path = self.parent.get_path()
        path.append((self.query, self.answer))
        return path


class MCTSSubqueryGenerator:
    """
    Initialize the MCTS Subquery Generator.
    @param root_query: original question to be answered
    @param root_answer: answer to the original question
    @param reward_method: method to evaluate the reward of the nodes,
                          allowed values: "llm", "em", "em_relaxed", jaccard", "accuracy", "f1"
    @param max_iterations: maximum number of iterations for MCTS
    @param branch_factor: number of subqueries to generate for each node
    """

    def __init__(self, root_query, root_answer, reward_method, max_iterations=10, branch_factor=2):
        self.max_iterations = max_iterations
        self.root_query = root_query
        self.root_answer = root_answer
        self.branch_factor = branch_factor
        self.reward_method = reward_method

    def select(self, node):
        """
        Select a node to expand using UCT.
        """
        current = node
        while len(current.children) > 0:
            current = current.UCT()
            if current is None:
                break
        return current

    def expand(self, node, branch_factor, simulation=False):
        """
        Generate possible subqueries for the given node.
        """
        if node.prev_subqueries:
            subqueries_text = ", ".join([f'"{sq}"' for sq in node.prev_subqueries])
        else:
            subqueries_text = ""
        prompt = f"""You are given a complex question and an evolving list of direct subquestions aimed at resolving it step by step.
If the original question requires no steps to be answered, return "<complete>"!
Otherwise, your task is to simplify the question by generating the next logical direct subquestion in the sequence.
- If no subquestions are provided, generate the first one to begin decomposition.
- If subquestions are already listed, generate the next needed to move toward an answer. You may use answers provided.
- If last subquestion directly answers the original, return "<complete>".

Example:  
Question: "Are there more people living in the capital of Spain or in the capital of Italy?"  
Subquestions: ["What is the capital of Spain?", "What is the capital of Italy?"]  
Next subquestion: "Which city has a larger population: Madrid or Rome?"

Now continue:
Question: {self.root_query}  
Subquestions and their answers: [{subqueries_text}]  
Next subquestion: ...
***GIVE ONLY THE SUBQUESTION!***"""
        response = run_ollama(prompt, model)

        # check for too complex subquestions
        cnt = 0
        while len(response) > 150 and cnt < 10:
            print(f"Subquery too complex, retrying... {cnt}", flush=True)
            response = run_ollama(prompt, model)
            cnt += 1

        # if node is terminal, answer the subquery and evaluate the reward
        if "<complete>" in response:
            node.answer = run_search_llm(node.query, tokenizer_search, model_search, device_search)
            self.evaluate_reward(node) 
            return []

        # otherwise, create couple of alternative subqueries
        subqueries = [response.strip()]

        for _ in range(self.branch_factor - 1):
            response = run_ollama(prompt, model)
            if (response and response.strip()            # subquery is properly generated
                and response.strip() not in subqueries   # subquery does not repeat
                and "<complete>" not in response         # not terminal
            ):       
                subqueries.append(response.strip())

        print('################# Generated subqueries ##################', flush=True)
        print(subqueries, flush=True)

        if not simulation:
            for subquery in subqueries:
                child_answer = run_search_llm(subquery, tokenizer_search, model_search, device_search)
                child_prev_subqueries = node.prev_subqueries + [f"Q:{subquery}, A:{child_answer}"]
                child = node.add_child(subquery, child_prev_subqueries)
                child.answer = child_answer
            return node.children
        else:
            fake_child_answer = run_search_llm(subqueries[0], tokenizer_search, model_search, device_search)
            fake_child_prev_subqueries = node.prev_subqueries + [f"Q:{subqueries[0]}, A:{fake_child_answer}"]
            fake_child = MCTSNode(subqueries[0], parent=node, prev_subqueries=fake_child_prev_subqueries)
            fake_child.answer = fake_child_answer
            return fake_child

    def simulate(self, node):
        """
        Simulate the MCTS rollout by expanding the node until a terminal node is reached.
        """
        next_node = node
        while next_node.answer is None:
            next_node = self.expand(next_node, 1, simulation=True)
        self.evaluate_reward(next_node)
        node.reward = next_node.reward

    def evaluate_reward(self, node):
        """
        Evaluate value of the given node.
        """
        if self.reward_method == "llm":
            score =reward_function(self.reward_method, node.answer, self.root_answer, self.root_query)
        else:
            score = reward_function(self.reward_method, node.answer, self.root_answer)
        node.reward = score
        return score

    def backpropagation(self, node, reward):
        """
        Update values of all nodes from the given node to the root.
        """
        current = node.parent
        while current is not None:
            current.update(reward)
            current = current.parent

    def run_mcts_single_query(self, query, filename=None):
        """
        Run the MCTS and return the best path found.
        @param query: original question to be answered
        @param filename: name of the file to save the visualization of the MCTS tree
        """
        root = MCTSNode(query)
        non_terminal_nodes = 1

        for iter in range(self.max_iterations):
            print(f"\n\n########################################## Iteration #{iter} ##########################################\n\n", flush=True)
            # Selection
            selected_node = self.select(root)
            non_terminal_nodes -= 1
            print(f"Selected node: {selected_node.query}", flush=True)

            # Expansion
            new_nodes = self.expand(selected_node, self.branch_factor)
            non_terminal_nodes += len(new_nodes)

            if new_nodes:
                # Simulation
                selected_node = random.choice(new_nodes)
                self.simulate(selected_node)
                print(f"Testing: #{iter} reward: {selected_node.reward}", flush=True)

                # Backpropagation
                self.backpropagation(selected_node, selected_node.reward)
            # early stoppings
            else:
                # if found terminal node with high reward
                if (selected_node.reward / max(1, selected_node.visits) > 0.8):
                    break
                # if all nodes are terminal
                if non_terminal_nodes == 0:
                    break
        
        if filename:
            graph = Digraph(comment="MCTS Tree", format="pdf")
            self._build_tree(graph, root)
            graph.render(filename, view=False)

        # find the best path
        best_terminal_node = self._dfs_best_terminal_node(root)[0]

        if best_terminal_node:
            path_data = best_terminal_node.get_path()
            final_answer = best_terminal_node.answer

            return {
                "original_query": query,
                "subqueries": path_data,
                "final_answer": final_answer,
                "confidence": best_terminal_node.reward,
            }
        else:
            return {
                "original_query": query,
                "subqueries": [],
                "final_answer": "Could not generate a reliable answer",
                "confidence": 0.0,
            }

    def _dfs_best_terminal_node(self, node, best_node=None, best_score=-float("inf")):
        """
        Run depth-first search to find the best terminal node with the highest reward/visits ratio.
        """
        if len(node.children) == 0 and node.visits > 0:
            score = node.reward / node.visits
            if score > best_score:
                best_node = node
                best_score = score

        for child in node.children:
            best_node, best_score = self._dfs_best_terminal_node(
                child, best_node, best_score
            )

        return best_node, best_score

    def _build_tree(self, graph, node, parent_id=None, counter=[0]):
        """
        Build the MCTS tree for visualization.
        """
        node_id = f"node{counter[0]}"
        label = f"Q: {node.query}\\nA: {node.answer or ''}\\nVisits: {node.visits}\\nReward: {node.reward:.2f}"
        graph.node(node_id, label=label, shape="box", style="rounded,filled", fillcolor="lightblue")
        counter[0] += 1

        if parent_id is not None:
            graph.edge(parent_id, node_id)

        current_id = node_id
        for child in node.children:
            self._build_tree(graph, child, current_id, counter)


# tokenizer, model, device = initialize_model("google/gemma-3-4b-it")
# time.sleep(60)  # wait for the model to load
model = 'qwen3:8b'
tokenizer_search, model_search, device_search = initialize_model("PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.2")
time.sleep(60)  # wait for the model to load

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str, default="nojobid", help='SLURM Job ID')
parser.add_argument('--run_settings', type=str, default="single_query", help='Run settings: single_query or db_eval')
parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset for db_eval mode')
parser.add_argument('--data_size', type=int, default=None, help='Number of queries to process in db_eval mode, None processes all queries')
args = parser.parse_args()
job_id = args.job_id
run_settings = args.run_settings
data_path = args.data_path
data_size = args.data_size

def main():
    if run_settings == "single_query":
        print(f"### Strating MCTS Subquery Generator with model {model}...", flush=True)

        query = "Which film has the director born later, 'Life Hits' or 'It's In The Air'?"
        answer = "It's In The Air"
        mcts = MCTSSubqueryGenerator(query, answer, max_iterations=10, reward_method="em_relaxed")
        result = mcts.run_mcts_single_query(query, filename=f"visualizations/approach2/mcts_tree_{job_id}")

        print(f"Original Query: {result['original_query']}", flush=True)
        print("\nSubquery Path:", flush=True)
        for i, (subquery, answer) in enumerate(result["subqueries"]):
            print(f"{i+1}. Q: {subquery}", flush=True)
            if answer:
                print(f"   A: {answer}", flush=True)

        print(f"\nFinal Answer: {result['final_answer']}", flush=True)
        print(f"Confidence: {result['confidence']}", flush=True)
    
    elif run_settings == "db_eval":
        print(f"### Starting MCTS Subquery Generator with model {model}...", flush=True)

        df = pd.read_parquet(data_path)
        qa_dataset = df.to_dict(orient='records')

        dataset_result = []
        output_path = f"json_results/mcts_results_{job_id}.jsonl"
        with open(output_path, "w") as f:
            for item in qa_dataset[:data_size]:
                query = item["question"]
                answer = item['golden_answers'][0]

                mcts = MCTSSubqueryGenerator(
                    query,
                    answer,
                    max_iterations=10,
                    branch_factor=2,
                    reward_method="jaccard"
                )
                result = mcts.run_mcts_single_query(query, filename=None)
                dataset_result.append({
                    "original_query": result["original_query"],
                    "true_answer": answer,
                    "subqueries": result["subqueries"],
                    "final_answer": result["final_answer"],
                    "confidence": result["confidence"],
                })

                f.write(json.dumps(dataset_result[-1]) + "\n")
                f.flush()
        
        print(f"### Average performance: {sum([item['confidence'] for item in dataset_result]) / len(dataset_result)}", flush=True)
        print("### All queries processed.", flush=True)

    else:
        print("Invalid run_settings parameter. Please use 'single_query' or 'db_eval'.", flush=True)


if __name__ == "__main__":
    main()
