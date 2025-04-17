import os
import random
import math
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from infer import initialize_model, run_llm, run_search_llm


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

        self.visits = 0
        self.reward = 0.0

    def add_child(self, child_query, child_subqueries):
        self.children.append(MCTSNode(child_query, parent=self, prev_subqueries=child_subqueries))
        return child

    def update(self, reward):
        self.visits += 1
        self.reward += reward

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
    def __init__(self, root_query, max_iterations=100, branch_factor=3):
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
        prompt = f"""You are given a multi-turn question that should be split into multiple single-turn subquestions. 
If subquestions are given between [ ], give the next subquestion. Otherwise, give a first subquestion. 
For example, for a question "How many people live in the capital of Spain" and subquestion "What is the capital of Spain?", 
the next subquestion is "How many people live in Madrid?". 
If no more subquestions are needed to describe the question, answer "<complete>". 
Question: {node.query}. 
Subquestions: [{subqueries_text}]."""

        response = run_llm(prompt, tokenizer, model, device)

        # if node is terminal, answer the subquery
        if "<complete>" in response:
            node.answer = run_search_llm(f"Answer this question concisely: {subquery}", tokenizer, model, device)
            return []

        # otherwise, create couple of alternative subqueries
        subqueries = [response.strip()]

        for _ in range(self.branch_factor - 1):
            response = run_llm(prompt, tokenizer, model, device)
            if (response and response.strip()            # subquery is properly generated
                and response.strip() not in subqueries   # subquery does not repeat
                and "<complete>" not in response         # not terminal
            ):       
                subqueries.append(response.strip())

        for subquery in subqueries:
            child_prev_subqueries = node.prev_subqueries + [subquery]
            child = node.add_child(subquery, child_prev_subqueries)
            child.answer = run_search_llm(f"Answer this question concisely: {subquery}", tokenizer, model, device)

        return subqueries

    def evaluate_reward(self, node):
        """
        Evaluate value of the given node.
        """
        prompt = f"""On a scale from 0.0 to 1.0, rate how relevant and helpful the following subquery and its answer are for answering the original question. Give only the numerical score.
Original Question: {self.root_query}
Subquery: {node.query}
Answer to Subquery: {node.answer}
Score (0.0-1.0): """
        response = run_llm(prompt, tokenizer, model, device)

        try:
            score = float(response.split("Score:")[-1].strip())
            if 0 <= score <= 1:
                return score
            else:
                return 0.5  # default score if not properly parsed
        except:
            return 0.5      # default score if parsing fails

    def backpropagation(self, node, reward):
        """
        Update values of all nodes from the given node to the root.
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def run_mcts(self, query):
        """
        Run the MCTS and return the best path found.
        """
        root = MCTSNode(query)

        for iter in range(self.max_iterations):
            # Selection
            print(f"Testing: starting node selection #{iter}", flush=True)
            selected_node = self.select(root)

            # Expansion
            print(f"Testing: starting node expansion #{iter}", flush=True)
            if len(selected_node.children) > 0:
                new_nodes = self.expand(selected_node)

                # If we expanded nodes, select one for simulation
                if new_nodes:
                    selected_node = random.choice(new_nodes)

            # Backpropagation
            print(f"Testing: starting reward backpropagation #{iter}", flush=True)
            self.backpropagation(selected_node, self.evaluate_reward(selected_node))

            # early stopping
            if (len(selected_node.children) > 0
                and selected_node.reward / max(1, selected_node.visits) > 0.8
            ):
                break

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


tokenizer, model, device = initialize_model()

def main():
    query = "Which film has the director born later, Life Hits or It's In The Air?"
    mcts = MCTSSubqueryGenerator(query, max_iterations=100)
    print("Testing: starting MCTS", flush=True)
    result = mcts.run_mcts(query)

    print(f"Original Query: {result['original_query']}", flush=True)
    print("\nSubquery Path:", flush=True)
    for i, (subquery, answer) in enumerate(result["subqueries"]):
        print(f"{i+1}. Q: {subquery}", flush=True)
        print(f"   A: {answer}", flush=True)

    print(f"\nFinal Answer: {result['final_answer']}", flush=True)
    print(f"Confidence: {result['confidence']:.2f}", flush=True)


if __name__ == "__main__":
    main()
