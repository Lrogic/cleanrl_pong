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

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

from cleanrl_utils.atari_wrappers import SaveOriginalObservation

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




def pca_trajectories(trajs, n_components=2):
    # trajs: list of trajectories, each is list of (D,) arrays
    X = np.concatenate([np.stack(traj, axis=0) for traj in trajs if len(traj) > 0], axis=0)  # (N, D)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xs)
    
    # Calculate reconstruction error (MSE on the scaled inputs)
    X_reconstructed = pca.inverse_transform(Z)
    reconstruction_error = np.mean((Xs - X_reconstructed) ** 2)

    print("explained_variance_ratio_", pca.explained_variance_ratio_)
    # print("singular_values_", pca.singular_values_)
    print("reconstruction_error_", reconstruction_error)

    # split back into trajectories
    out = []
    idx = 0
    # Also compute L2 norms between consecutive original vectors for each trajectory
    norms_per_traj = []
    for traj in trajs:
        T = len(traj)
        out.append(Z[idx:idx+T])
        idx += T

        # compute L2 norms on the original vectors (not PCA-projected)
        if T > 1:
            arr = np.stack(traj, axis=0)
            diffs = arr[1:] - arr[:-1]
            norms = np.linalg.norm(diffs, axis=1)
            norms_per_traj.append(norms)
        else:
            norms_per_traj.append(np.array([]))

    return out, reconstruction_error, norms_per_traj


def plot_pca_trajectories(pca_traj, reconstruction_error=None, output_dir="pca_plots",
                          num_frames_to_plot=10, l2_norms=None, frames=None):
    # NOTE: This function expects a single PCA trajectory (an (T,2) array) as
    # `pca_traj`. If you have multiple trajectories (a list), pass a single
    # element (e.g. `pca_trajs[0]`) when calling this function.
    #
    # `num_frames_to_plot` controls how many points/images to display. If
    # positive, the first `num_frames_to_plot` states are shown. If negative,
    # the last `abs(num_frames_to_plot)` states are shown (e.g. -10 -> last 10).
    os.makedirs(output_dir, exist_ok=True)
    # create a side-by-side figure: left PCA, right image grid
    fig, (ax_pca, ax_imgs) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})

    # determine slice indices based on num_frames_to_plot (supports negative)
    full_traj = np.array(pca_traj)
    T = len(full_traj)
    if num_frames_to_plot >= 0:
        k = min(num_frames_to_plot, T)
        start_idx = 0
    else:
        k = min(abs(num_frames_to_plot), T)
        start_idx = max(0, T - k)

    traj = full_traj[start_idx:start_idx + k]
    
    # Plot points at each timestep on PCA axis
    ax_pca.scatter(traj[:, 0], traj[:, 1], s=50, alpha=0.7, zorder=5)
    
    # Highlight the first point with a different color
    ax_pca.scatter(traj[0, 0], traj[0, 1], s=100, color='green', alpha=0.9, zorder=6, label='Start')
    # Highlight the last point with a different color
    ax_pca.scatter(traj[-1, 0], traj[-1, 1], s=100, color='red', alpha=0.9, zorder=6, label='End')
    
    # L2 norms for this trajectory (if provided)
    l2_first = np.array(l2_norms) if l2_norms is not None else None
    # slice norms to match displayed points: norms length is T-1
    if l2_first is not None:
        l2_first = l2_first[start_idx:start_idx + max(0, k - 1)]

    # Add arrows between consecutive points
    for i in range(len(traj) - 1):
        x_start, y_start = traj[i, 0], traj[i, 1]
        x_end, y_end = traj[i + 1, 0], traj[i + 1, 1]
        dx = x_end - x_start
        dy = y_end - y_start
        ax_pca.arrow(x_start, y_start, dx, dy, 
                 head_width=0.05, head_length=0.05, fc='black', ec='black', alpha=0.6, zorder=4)
        
        # Add label at midpoint of arrow: use L2 norm if available, otherwise index
        mid_x = x_start + dx / 2
        mid_y = y_start + dy / 2
        if l2_first is not None and i < len(l2_first):
            label_text = f"{l2_first[i]:.1f}"
        else:
            label_text = str(i+1)
        ax_pca.text(mid_x, mid_y, label_text, fontsize=8, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add index labels above each PCA point (use a different color than L2 labels)
    # try:
    #     y_min, y_max = ax_pca.get_ylim()
    #     y_offset = (y_max - y_min) * 0.03
    # except Exception:
    #     y_offset = 0.02
        for j in range(len(traj)):
            px, py = traj[j, 0], traj[j, 1]
            idx_label = str(j)
            ax_pca.annotate(
                idx_label,
                xy=(px, py),
                xytext=(0, 6),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color='black',
                fontsize=8,
            )

    title = "PCA of CNN Feature Trajectories"
    if reconstruction_error is not None:
        title += f"\nReconstruction Error: {reconstruction_error:.4f}"
    
    ax_pca.set_title(title)
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    ax_pca.legend()
    ax_pca.grid(True)

    # Right: image grid. Use the provided `frames` list for this trajectory.
    # `frames` is expected to be a list of frame stacks shape (4, H, W); use the
    # first channel (index 0) of each stack as the grayscale image to display.
    if frames is not None:
        # slice frames to match displayed points
        full_frames = list(frames)
        frames_to_use = full_frames[start_idx:start_idx + k]
        n = len(frames_to_use)
        if n > 0:
            imgs = []
            for i in range(n):
                stack = np.array(frames_to_use[i])  # (4, H, W)
                img = stack[0]  # use first frame from the stack
                # normalize to 0-255 if floats
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                imgs.append(img)

            # build mosaic
            cols = min(n, 5)
            rows = int(np.ceil(n / cols))
            H, W = imgs[0].shape
            mosaic = np.ones((rows * H, cols * W), dtype=np.uint8) * 255
            for idx in range(n):
                r = idx // cols
                c = idx % cols
                mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = imgs[idx]

            ax_imgs.imshow(mosaic, cmap='gray', vmin=0, vmax=255)
            ax_imgs.set_xticks([])
            ax_imgs.set_yticks([])

            # add index above each image (top-center inside image)
            for idx in range(n):
                r = idx // cols
                c = idx % cols
                x = c * W + W / 2
                y = r * H + 6
                ax_imgs.text(x, y, str(idx), color='white', fontsize=9, ha='center', va='top',
                              bbox=dict(facecolor='black', alpha=0.6, pad=1))
    else:
        ax_imgs.text(0.5, 0.5, 'No frames provided', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_trajectory.png"))
    plt.show()

def plot_grayscale_frame(frame):
    frame = frame[0]
    plt.imshow(frame)
    plt.title("Grayscale Frame")
    plt.axis('off')
    plt.show()


def get_wrapper(env, wrapper_type):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_type):
            return env
        env = env.env
    return None


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
    model.network[7].register_forward_hook(get_activation('cnn_features'))


    obs, _ = envs.reset()
    episodic_returns = []
    cnn_list = []
    frames = []

    if run_rrt:
        run_rrt_exploration(model,
            envs,
            device,
            num_iterations = 100,
            expansion_steps = 1,
            best_action_prob = 1.0)
        return
    
    trajectories = []
    cur_traj = []
    cur_traj_frames = []

    while len(episodic_returns) < eval_episodes:
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values, best_action_prob=1.0)

        cnn_fts = activation['cnn_features']
        cnn_fts = cnn_fts.detach().cpu().clone().numpy().squeeze()
        cur_traj.append(cnn_fts)

        # plot_grayscale_frame(obs[0])  # visualize the original observation frame for this step
        # return
        cur_traj_frames.append(obs[0])  # save the original observation frame for this step
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if rewards[0] != 0:
            trajectories.append(cur_traj[:])
            cur_traj = []
            frames.append(cur_traj_frames[:])
            cur_traj_frames = []
            break

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    
    if do_frame_cosine_similarity:
        cnn_similarity(cnn_list, num_tests=100)

    pca_trajs, reconstruction_error, l2_norms = pca_trajectories(trajectories, n_components=2)
    # Pass a single PCA trajectory and its corresponding L2 norms to the plot
    single_l2 = l2_norms[0] if (l2_norms is not None and len(l2_norms) > 0) else None
    single_frames = frames[0] if (frames is not None and len(frames) > 0) else None
    plot_pca_trajectories(pca_trajs[0], reconstruction_error=None, 
                          l2_norms=single_l2, frames=single_frames, num_frames_to_plot=-10)
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
        do_frame_cosine_similarity=False,
        run_rrt=False
    )
