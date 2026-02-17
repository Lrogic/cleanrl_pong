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
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from typing import List

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


from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def compute_MDS(embeddings: List[np.ndarray], similarity_metric: str = "euclidean"):
    X = np.asarray(embeddings, dtype=np.float64)

    # Pairwise dissimilarities in original space
    D = pairwise_distances(X, metric=similarity_metric)

    metric = True
    normalized_stress = False

    print(f"metric: {metric}, similarity_measure: {similarity_metric}, normalized: {normalized_stress}")

    mds = MDS(
        n_components=2,
        metric=metric,                 # metric MDS
        dissimilarity="precomputed", # using precomputed distance matrix D
        n_init=8,                    # multiple restarts
        max_iter=1000,
        random_state=0,
        normalized_stress=normalized_stress,
    )

    X_transformed = mds.fit_transform(D)  # shape (n, 2)

    # --- Preservation metrics ---
    D_2d = squareform(pdist(X_transformed, metric="euclidean"))  # pairwise distances in 2D
    mask = np.triu_indices_from(D, k=1)                          # unique pairs only

    pearson_r = pearsonr(D[mask], D_2d[mask]).statistic
    spearman_rho = spearmanr(D[mask], D_2d[mask]).statistic
    rmse_distance = np.sqrt(np.mean((D[mask] - D_2d[mask]) ** 2))

    print(f"Stress: {mds.stress_}")
    print(f"Pearson r (distance preservation): {pearson_r:.4f}")
    print(f"Spearman rho (rank preservation):  {spearman_rho:.4f}")
    print(f"RMSE (distance error):             {rmse_distance:.4f}")

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=30)

    # Highlight first and last points
    plt.scatter(
        X_transformed[0, 0], X_transformed[0, 1],
        s=100, color="green", alpha=0.9, zorder=6, label="Start"
    )
    plt.scatter(
        X_transformed[-1, 0], X_transformed[-1, 1],
        s=100, color="red", alpha=0.9, zorder=6, label="End"
    )

    for i, (x, y) in enumerate(X_transformed):
        plt.text(x, y, str(i), fontsize=8)

    plt.axis("equal")
    plt.title(f"Metric MDS (2D) | metric={similarity_metric}")
    plt.legend()
    plt.show()

    return X_transformed, {
        "stress": float(mds.stress_),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "rmse_distance": float(rmse_distance),
    }


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.network[7].register_forward_hook(get_activation('cnn_features'))


    obs, _ = envs.reset()
    episodic_returns = []
    
    trajs = []
    cur_traj = []

    while len(episodic_returns) < eval_episodes:
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values, best_action_prob=1.0)

        cnn_fts = activation['cnn_features']
        cnn_fts = cnn_fts.detach().cpu().clone().numpy().squeeze()
        cur_traj.append(cnn_fts)
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if rewards[0] != 0:
            trajs.append(cur_traj[:])
            cur_traj = []
            break

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    
    compute_MDS(trajs[0], "euclidean")

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn_atari import QNetwork, make_env, get_activation, activation, restore_env

    # model_path = hf_hub_download(repo_id="cleanrl/PongNoFrameskip-v4-dqn_atari-seed1", filename="dqn_atari.cleanrl_model")
    model_path = hf_hub_download(repo_id="cleanrl/PongNoFrameskip-v4-dqn_atari-seed1", filename="dqn_atari.cleanrl_model", local_files_only=True)
    
    evaluate(
        model_path,
        make_env,
        "PongNoFrameskip-v4",
        eval_episodes=1,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )
