import os
import random
import math
import json
import requests
import time
from graphviz import Digraph
from typing import List, Dict, Any, Optional, Tuple
import argparse
from infer import initialize_model, run_ollama, run_llm, run_search_llm


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

    def best_child(self, c_param=2):
        """
        Select the child node with the highest Upper Confidence Bound (UCB) value.
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
        if self.answer:
            path.append((self.query, self.answer))
        return path


class MCTSSubqueryGenerator:
    def __init__(self, root_query, max_iterations=10, branch_factor=2):
        self.max_iterations = max_iterations
        self.root_query = root_query
        self.branch_factor = branch_factor

    def select(self, node):
        """
        Select a node to expand using UCB.
        """
        current = node
        while len(current.children) > 0:
            current = current.best_child()
            if current is None:
                break
        return current

    def expand(self, node):
        """
        Generate possible subqueries for the given node.
        """
        if node.prev_subqueries:
            subqueries_text = ", ".join([f'"{sq}"' for sq in node.prev_subqueries])
        else:
            subqueries_text = ""
        prompt = f"""You are given a complex question and an evolving list of direct subquestions aimed at resolving it step by step.
Your task is to simplify the question by generating the **next logical direct subquestion** in the sequence.
- If no subquestions are provided, generate the **first** one to begin decomposition.
- If subquestions are already listed, generate the **next** needed to move toward an answer.
- If the original question is fully resolved by the last subquestion, return "<complete>".

Example:  
Question: "Are there more people living in the capital of Spain or in the capital of Italy?"  
Subquestions: ["What is the capital of Spain?", "What is the capital of Italy?"]  
Next subquestion: "Which city has a larger population: Madrid or Rome?"

Now continue:
Question: {self.root_query}  
Subquestions: [{subqueries_text}]  
Next subquestion: ...
***GIVE ONLY THE SUBQUESTION!***"""

        print("test1", flush=True)

        response = run_ollama(prompt, model)

        # check for too complex subquestions
        cnt = 0
        while len(response) > 150 and cnt < 10:
            response = run_ollama(prompt, model)
            cnt += 1

        # if node is terminal, answer the subquery
        if "<complete>" in response:
            node.answer = run_search_llm(node.query, tokenizer_search, model_search, device_search)
            return []

        # otherwise, create couple of alternative subqueries
        subqueries = [response.strip()]

        print("test2", flush=True)

        for _ in range(self.branch_factor - 1):
            response = run_ollama(prompt, model)
            if (response and response.strip()            # subquery is properly generated
                and response.strip() not in subqueries   # subquery does not repeat
                and "<complete>" not in response         # not terminal
            ):       
                subqueries.append(response.strip())
            print("test_sth", flush=True)

        print('################# Generated subqueries ##################', flush=True)
        print(subqueries, flush=True)

        for subquery in subqueries:
            child_prev_subqueries = node.prev_subqueries + [subquery]
            child = node.add_child(subquery, child_prev_subqueries)
            child.answer = run_search_llm(subquery, tokenizer_search, model_search, device_search)

        return node.children

    def evaluate_reward(self, node):
        """
        Evaluate value of the given node.
        """
        prompt = f"""On a scale from 0.0 to 1.0, rate how relevant and helpful the following subquery and its answer are for answering the original question. Give only the numerical score.
Original Question: {self.root_query}
Subquery: {node.query}
Answer to Subquery: {node.answer}
Score (0.0-1.0): """
        response = run_ollama(prompt, model)

        try:
            score = float(response.split("Score:")[-1].strip())
            if 0 <= score <= 1:
                node.reward = score
            else:
                node.reward = 0.5 # default score if not properly parsed
        except:
            node.reward = 0.5 # default score if parsing fails

    def backpropagation(self, node, reward):
        """
        Update values of all nodes from the given node to the root.
        """
        current = node.parent
        while current is not None:
            current.update(reward)
            current = current.parent

    def run_mcts(self, query, filename="mcts_tree"):
        """
        Run the MCTS and return the best path found.
        @param query: original question to be answered
        @param filename: name of the file to save the visualization of the MCTS tree
        """
        root = MCTSNode(query)

        for iter in range(self.max_iterations):
            print(f"\n\n########################################## Iteration #{iter} ##########################################\n\n", flush=True)
            # Selection
            selected_node = self.select(root)
            print(f"Selected node: {selected_node.query}", flush=True)

            # Expansion
            new_nodes = self.expand(selected_node)

            # If we expanded nodes, select one for simulation
            if new_nodes:
                selected_node = random.choice(new_nodes)

            # Backpropagation
            self.evaluate_reward(selected_node)
            print(f"Testing: #{iter} reward: {selected_node.reward}", flush=True)
            self.backpropagation(selected_node, selected_node.reward)

            # early stopping
            if (len(selected_node.children) > 0
                and selected_node.reward / max(1, selected_node.visits) > 0.8
            ):
                break
        
        graph = Digraph(comment="MCTS Tree", format="pdf")
        self._build_tree(graph, root)
        graph.render(filename, view=False)

        # find the best path
        best_path = self._dfs_best_terminal_node(root)[0]

        if best_path:
            path_data = best_path.get_path()
            final_answer = best_path.answer

            return {
                "original_query": query,
                "subqueries": path_data,
                "final_answer": final_answer,
                "confidence": best_path.reward / max(1, best_path.visits),
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
args = parser.parse_args()
job_id = args.job_id

def main():
    print(f"### Strating MCTS Subquery Generator with model {model}...", flush=True)

    query = "Which film has the director born later, 'Life Hits' or 'It's In The Air'?"
    mcts = MCTSSubqueryGenerator(query, max_iterations=3)
    result = mcts.run_mcts(query, filename=f"visualizations/approach1/mcts_tree_{job_id}")

    print(f"Original Query: {result['original_query']}", flush=True)
    print("\nSubquery Path:", flush=True)
    for i, (subquery, answer) in enumerate(result["subqueries"]):
        print(f"{i+1}. Q: {subquery}", flush=True)
        print(f"   A: {answer}", flush=True)

    print(f"\nFinal Answer: {result['final_answer']}", flush=True)
    print(f"Confidence: {result['confidence']:.2f}", flush=True)


if __name__ == "__main__":
    main()
