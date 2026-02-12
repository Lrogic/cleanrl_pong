import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import statistics
from pprint import pprint
import copy
import pickle
import sys
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

def save_envs(envs):
    return [env.unwrapped.ale.cloneSystemState() for env in envs]

def restore_envs(env_id, saved_env_states, idx, capture_video, run_name):
    return gym.vector.SyncVectorEnv([restore_env(env_id, saved_env_state, idx, capture_video, run_name) for 
            saved_env_state in saved_env_states])

# get the action from the q_values
# pick either the best action (as judged by the Q-network) with prob=best_action_prob, 
# or second best action with prob=(1 - best_action_prob)
def get_actions(q_values, best_action_prob):
    top_k = torch.topk(q_values, k=2).indices.cpu().numpy().squeeze()
    idx = 1 if (random.random() > best_action_prob) else 0
    return [top_k[idx]]

def cnn_similarity(cnn_list, num_tests=30):
    n = len(cnn_list)
    # grab two consecutive features. we expect them to be close
    consecutive_vals = []
    print("------------ CONSECUTIVE FARMES TEST ------------")
    for i in range(num_tests):
        idx = random.randint(0, n-2)
        cos = torch.nn.CosineSimilarity()
        sim = cos(cnn_list[idx], cnn_list[idx+1]).item()
        consecutive_vals.append(sim)
        # print(f"Cosine similarity between feature at index {idx} and index {idx+1}: {sim}")
    print(f"Average cosine similarity between consecutive frames: {statistics.mean(consecutive_vals)}")
    # grab two random features. we expect them to be further
    random_vals = []
    print("------------ RANDOM FRAMES TEST ------------")
    for i in range(num_tests):
        idx1 = random.randint(0, n-1)
        idx2 = random.randint(0, n-1)
        cos = torch.nn.CosineSimilarity()
        sim = cos(cnn_list[idx1], cnn_list[idx2]).item()
        random_vals.append(sim)
        # print(f"Cosine similarity between feature at index {idx1} and index {idx2}: {sim}")
    print(f"Average cosine similarity between random frames: {statistics.mean(random_vals)}")


@dataclass
class RRTNode:
    """Represents a node in the RRT graph"""
    node_id: int
    obs: np.ndarray  # observation from the environment
    cnn_features: torch.Tensor  # CNN features for this state
    ale_state: object  # saved ALE system state for reproducibility


class RRTGraph:
    """RRT graph structure for exploring state space using CNN features"""
    
    def __init__(self, device="cpu", cosine_sim_threshold=0.95):
        self.graph = nx.DiGraph()  # Directed graph for parent->child relationships
        self.nodes_data = {}  # node_id -> RRTNode
        self.next_node_id = 0
        self.device = device
        # self.cosine_sim_threshold = cosine_sim_threshold
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
    
    def add_node(self, obs, cnn_features, ale_state):
        """Add a node to the graph"""
        node_id = self.next_node_id
        node = RRTNode(
            node_id=node_id,
            obs=obs.copy() if isinstance(obs, np.ndarray) else obs,
            cnn_features=cnn_features.detach().cpu().clone(),
            ale_state=ale_state # do we have to copy this?
        )
        self.nodes_data[node_id] = node
        self.graph.add_node(node_id)
        self.next_node_id += 1
        return node_id
    
    def add_edge(self, parent_id, child_id, action):
        """Add an edge (action) between two nodes"""
        self.graph.add_edge(parent_id, child_id, action=action)
    
    def find_nearest_neighbor(self, query_features):
        """
        Find the nearest node to query_features using cosine similarity.
        Returns: (node_id, similarity_score)
        """
        if not self.nodes_data:
            return None, None
        
        # Stack all CNN features
        all_features = torch.stack([node.cnn_features for node in self.nodes_data.values()])
        query_features_expanded = query_features.unsqueeze(0)
        
        # Compute cosine similarities
        similarities = self.cosine_sim(query_features_expanded, all_features)
        
        nearest_idx = torch.argmax(similarities).item()
        nearest_node_id = list(self.nodes_data.keys())[nearest_idx]
        similarity_score = similarities[nearest_idx].item()
        
        return nearest_node_id, similarity_score
    
    def get_all_features(self):
        """Get all CNN features as a stacked tensor for analysis"""
        if not self.nodes_data:
            return None
        return torch.stack([node.cnn_features for node in self.nodes_data.values()])
    
    def get_node(self, node_id):
        """Get node data by ID"""
        return self.nodes_data.get(node_id)
    
    def get_successors(self, node_id):
        """Get all child nodes"""
        return list(self.graph.successors(node_id))
    
    def get_predecessors(self, node_id):
        """Get all parent nodes"""
        return list(self.graph.predecessors(node_id))
    
    def get_edge_data(self, parent_id, child_id):
        """Get edge attributes (action) between two nodes"""
        return self.graph.get_edge_data(parent_id, child_id)
    
    def __len__(self):
        return len(self.nodes_data)


def run_rrt_exploration(
    model,
    envs,
    device,
    num_iterations: int = 1000,
    expansion_steps: int = 1,
    best_action_prob: float = 1.0,
):
    """
    Run RRT exploration algorithm on the environment.
    
    Args:
        model: DQN model with CNN features hooked
        envs: Vector environment
        device: torch device
        num_iterations: Number of RRT iterations to run
        expansion_steps: Number of steps to expand from each node (1 or more)
        best_action_prob: Probability of picking best action vs second best
    
    Returns:
        graph: RRTGraph containing all explored nodes and edges
    """
    graph = RRTGraph(device=device)
    
    # Initialize with current state
    obs, _ = envs.reset()
    q_values = model(torch.Tensor(obs).to(device))
    cnn_fts = activation['cnn_features']
    saved_state = save_envs(envs.envs)[0]
    
    root_id = graph.add_node(obs[0], cnn_fts.squeeze(0), saved_state)
    print(f"[RRT] Added root node: {root_id}")
    
    for iteration in tqdm(range(num_iterations)):
        # Sample a random state
        obs_random, _ = envs.reset()
        q_values_random = model(torch.Tensor(obs_random).to(device))
        cnn_fts_random = activation['cnn_features']
        
        # Find nearest neighbor
        nearest_id, similarity = graph.find_nearest_neighbor(cnn_fts_random.squeeze(0))
        
        # if iteration % 100 == 0:
        #     print(f"[RRT] Iteration {iteration}, Graph size: {len(graph)}, Nearest similarity: {similarity:.4f}")
        
        # Restore to nearest neighbor and expand
        saved_state_nearest = graph.nodes_data[nearest_id].ale_state
        envs_expanded = restore_envs(
            env_id=envs.envs[0].unwrapped.spec.id,
            saved_env_states=[saved_state_nearest],
            idx=0,
            capture_video=False,
            run_name="rrt_exploration"
        )
        
        current_parent_id = nearest_id
        
        # Expand from this node
        for step in range(expansion_steps):
            obs_expanded = graph.nodes_data[current_parent_id].obs
            obs_expanded_tensor = np.expand_dims(obs_expanded, axis=0)
            
            q_values_expanded = model(torch.Tensor(obs_expanded_tensor).to(device))
            actions = get_actions(q_values_expanded, best_action_prob=best_action_prob)
            cnn_fts_expanded = activation['cnn_features']
            
            next_obs, _, _, _, _ = envs_expanded.step(actions)
            q_values_next = model(torch.Tensor(next_obs).to(device))
            cnn_fts_next = activation['cnn_features']
            saved_state_next = save_envs(envs_expanded.envs)[0]
            
            # Add new node
            new_node_id = graph.add_node(next_obs[0], cnn_fts_next.squeeze(0), saved_state_next)
            graph.add_edge(current_parent_id, new_node_id, actions[0])
            
            current_parent_id = new_node_id
    
    print(f"[RRT] Exploration complete. Final graph size: {len(graph)}")
    return graph



def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    do_frame_cosine_similarity: bool = False,
    run_rrt: bool = False
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.network[8].register_forward_hook(get_activation('cnn_features'))


    obs, _ = envs.reset()
    episodic_returns = []
    cnn_list = []

    if run_rrt:
        run_rrt_exploration(model,
            envs,
            device,
            num_iterations = 100,
            expansion_steps = 1,
            best_action_prob = 1.0)
        return

    i = 0
    while len(episodic_returns) < eval_episodes:
        i += 1
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values, best_action_prob=1.0)
        cnn_fts = activation['cnn_features']
        if do_frame_cosine_similarity and (len(episodic_returns) == 0) and i > 500 and i < 1000:
            cnn_list.append(cnn_fts.detach().cpu().clone())
        next_obs, _, _, _, infos = envs.step(actions)
        # pprint(vars(envs.envs[0]))
        # return
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        # if i == 500:
        #     saved_envs = save_envs(envs.envs)
        #     envs2 = restore_envs(env_id, saved_envs, 0, capture_video, run_name)
        #     envs3 = copy.deepcopy(envs)
        #     envs4 = copy.deepcopy(envs.unwrapped)

        #     test1 = pickle.dumps(envs.envs)
        #     print(sys.getsizeof(test1))
        #     test2 = pickle.dumps(envs2.envs)
        #     print(sys.getsizeof(test2))
        #     test3 = pickle.dumps(envs3.envs)
        #     print(sys.getsizeof(test3))
        #     test4 = pickle.dumps(envs4.envs)
        #     print(sys.getsizeof(test4))

        #     x = 1
        # if i == 1000:
        #     print("saving and reloading envs")
        #     saved_envs = save_envs(envs.envs)
        #     envs.close()
        #     envs = restore_envs(env_id, saved_envs, 0, capture_video, run_name)
        obs = next_obs
    
    if do_frame_cosine_similarity:
        cnn_similarity(cnn_list, num_tests=100)
    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn_atari import QNetwork, make_env, get_activation, activation, restore_env

    model_path = hf_hub_download(repo_id="cleanrl/PongNoFrameskip-v4-dqn_atari-seed1", filename="dqn_atari.cleanrl_model")
    
    evaluate(
        model_path,
        make_env,
        "PongNoFrameskip-v4",
        eval_episodes=1,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
        do_frame_cosine_similarity=True,
        run_rrt=False
    )
