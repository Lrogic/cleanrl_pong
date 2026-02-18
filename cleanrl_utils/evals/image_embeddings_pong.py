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

from typing import List, Optional

import os
from PIL import Image
import clip

from cleanrl_utils.atari_wrappers import SaveOriginalObservation

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

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


def compute_MDS(embeddings: List[np.ndarray], similarity_metric: str = "euclidean", arrows=False, 
                frames: Optional[List] = None, num_frames_to_plot=10,
                save_path: Optional[str] = None, title: Optional[str] = None):
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

    # --- Feature consistency ---
    embeddings_tensor = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
    con_mean, rand_val = feature_consistency(embeddings_tensor)
    fc_text = f"consecutive_frame_mean={con_mean:.3f}, random_frame_mean={rand_val:.3f}"

    # --- Plot ---
    # Create figure with two subplots if frames are provided, otherwise single plot
    if frames is not None and len(frames) > 0:
        fig, (ax_mds, ax_imgs) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})
    else:
        fig = plt.figure(figsize=(8, 6))
        ax_mds = plt.gca()
        ax_imgs = None

    ax_mds.scatter(X_transformed[:, 0], X_transformed[:, 1], s=30)

    # Highlight first and last points
    ax_mds.scatter(
        X_transformed[0, 0], X_transformed[0, 1],
        s=100, color="green", alpha=0.9, zorder=6, label="Start"
    )
    ax_mds.scatter(
        X_transformed[-1, 0], X_transformed[-1, 1],
        s=100, color="red", alpha=0.9, zorder=6, label="End"
    )

    for i, (x, y) in enumerate(X_transformed):
        ax_mds.text(x, y, str(i), fontsize=8)

    # draw arrows between consecutive points and label with 2D distance
    if arrows:
        for i in range(len(X_transformed) - 1):
            x_start, y_start = X_transformed[i, 0], X_transformed[i, 1]
            x_end, y_end = X_transformed[i + 1, 0], X_transformed[i + 1, 1]
            dx = x_end - x_start
            dy = y_end - y_start
            # use annotate so arrow head size is specified in points (mutation_scale)
            ax_mds.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='black', alpha=0.6, mutation_scale=10),
                zorder=4,
            )
            mid_x = x_start + dx / 2
            mid_y = y_start + dy / 2
            # use original-space pairwise distance matrix `D` computed above
            dist_orig = float(D[i, i + 1])
            ax_mds.text(mid_x, mid_y, f"{dist_orig:.2f}", fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

    ax_mds.axis("equal")
    ax_mds.set_title(f"{title}\nMetric MDS (2D) | metric={similarity_metric}\n{fc_text}")
    ax_mds.legend()


    T = len(embeddings)
    if num_frames_to_plot >= 0:
        k = min(num_frames_to_plot, T)
        start_idx = 0
    else:
        k = min(abs(num_frames_to_plot), T)
        start_idx = max(0, T - k)

    # plot frames grid on the right if provided
    if ax_imgs is not None and len(frames) > 0:
        # select every Nth frame to avoid clutter (e.g., up to 10 frames total)
        # max_frames = 10
        # if len(frames) > max_frames:
        #     step = len(frames) // max_frames
        #     selected_indices = list(range(0, len(frames), step))[:max_frames]
        # else:
        #     selected_indices = list(range(len(frames)))
        selected_indices = list(range(start_idx, start_idx+k))

        imgs = []
        for idx in selected_indices:
            frame = frames[idx]
            # handle different frame formats (grayscale, RGB, frame stack)
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] == 4:
                    # frame stack (C, H, W) - use first channel
                    img = frame[0]
                elif frame.ndim == 3 and frame.shape[-1] == 3:
                    # RGB image (H, W, C)
                    img = frame
                elif frame.ndim == 2:
                    # grayscale (H, W)
                    img = frame
                else:
                    img = frame[0] if frame.ndim >= 3 else frame
            else:
                continue

            # normalize to 0-255 if needed
            if img.max() <= 1.0 and img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            imgs.append(img)

        if len(imgs) > 0:
            # build mosaic
            cols = min(len(imgs), 5)
            rows = int(np.ceil(len(imgs) / cols))
            if imgs[0].ndim == 2:
                # grayscale
                H, W = imgs[0].shape
                mosaic = np.ones((rows * H, cols * W), dtype=np.uint8) * 255
                for idx, img in enumerate(imgs):
                    r = idx // cols
                    c = idx % cols
                    mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = img
                ax_imgs.imshow(mosaic, cmap='gray', vmin=0, vmax=255)
            else:
                # RGB
                H, W = imgs[0].shape[:2]
                mosaic = np.ones((rows * H, cols * W, 3), dtype=np.uint8) * 255
                for idx, img in enumerate(imgs):
                    r = idx // cols
                    c = idx % cols
                    if img.ndim == 2:
                        # convert grayscale to RGB for mosaic
                        img_rgb = np.stack([img, img, img], axis=-1)
                    else:
                        img_rgb = img
                    mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = img_rgb
                ax_imgs.imshow(mosaic)

            ax_imgs.set_xticks([])
            ax_imgs.set_yticks([])

            # add frame indices on the mosaic
            for idx, selected_idx in enumerate(selected_indices):
                r = idx // cols
                c = idx % cols
                x = c * W + W / 2
                y = r * H + 6
                ax_imgs.text(x, y, str(selected_idx), color='white', fontsize=9, ha='center', va='top',
                              bbox=dict(facecolor='black', alpha=0.6, pad=1))

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    # else:
    #     plt.show()

    return X_transformed, {
        "stress": float(mds.stress_),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "rmse_distance": float(rmse_distance),
    }

def get_wrapper(env, wrapper_type):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_type):
            return env
        env = env.env
    return None

def feature_consistency(feat_list):
    n = len(feat_list)
    cos = torch.nn.CosineSimilarity(dim=0)
    # grab two consecutive features. we expect them to be close
    consecutive_vals = []
    for i in range(n-1):
        sim = cos(feat_list[i], feat_list[i+1]).item()
        consecutive_vals.append(sim)
    # grab two random features. we expect them to be far
    random_vals = []
    for i in range(n):
        idx1 = random.randint(0, n-1)
        idx2 = random.randint(0, n-1)
        sim = cos(feat_list[idx1], feat_list[idx2]).item()
        random_vals.append(sim)
    return np.mean(consecutive_vals), np.mean(random_vals)


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

    # normalize device (allow callers to pass string)
    if isinstance(device, str):
        device = torch.device(device)

    # load CLIP model on matching device type (cpu/cuda)
    clip_device_str = "cuda" if device.type == "cuda" and torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device_str)
    clip_torch_device = torch.device(clip_device_str)


    obs, _ = envs.reset()
    episodic_returns = []
    
    trajs_DQN = []
    cur_DQN_traj = []
    trajs_clip = []
    cur_clip_traj = []
    
    dqn_frames = []
    cur_dqn_frames = []
    clip_frames = []
    cur_clip_frames = []

    original_env = get_wrapper(envs.envs[0], SaveOriginalObservation)

    while len(episodic_returns) < eval_episodes:
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values, best_action_prob=1.0)

        cnn_fts = activation['cnn_features']
        cnn_fts = cnn_fts.detach().cpu().clone().numpy().squeeze()
        cur_DQN_traj.append(cnn_fts)
        # collect DQN frame (obs[0] is the stacked frame)
        cur_dqn_frames.append(obs[0])
        
        # collect original frame saved by SaveOriginalObservation wrapper
        try:
            orig = original_env.old_obs
        except Exception:
            orig = None

        if orig is not None:
            try:
                pil = Image.fromarray(orig)
                with torch.no_grad():
                    clip_input = clip_preprocess(pil).unsqueeze(0).to(clip_torch_device)
                    img_feat = clip_model.encode_image(clip_input)
                    clip_fts = img_feat.detach().cpu().numpy().squeeze()
            except Exception:
                clip_fts = np.zeros(512)
            # collect CLIP frame (original RGB)
            cur_clip_frames.append(orig)
        else:
            clip_fts = np.zeros(512)
            # collect placeholder if no original
            cur_clip_frames.append(np.zeros((84, 84, 3), dtype=np.uint8))

        cur_clip_traj.append(clip_fts)
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if rewards[0] != 0:
            trajs_DQN.append(cur_DQN_traj[:])
            cur_DQN_traj = []
            trajs_clip.append(cur_clip_traj[:])
            cur_clip_traj = []
            dqn_frames.append(cur_dqn_frames[:])
            cur_dqn_frames = []
            clip_frames.append(cur_clip_frames[:])
            cur_clip_frames = []
            break

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    
    # compute_MDS(trajs_DQN[0], "euclidean")
    # also compute and save MDS for CNN features and CLIP embeddings
    compute_MDS(trajs_DQN[0], "cosine", save_path=os.path.join("MDS_plots", "cnn_mds.png"), 
                title="DQN Embeddings", 
                num_frames_to_plot=-10, frames=dqn_frames[0] if len(dqn_frames) > 0 else None)
    compute_MDS(trajs_clip[0], "cosine", save_path=os.path.join("MDS_plots", "clip_mds.png"), 
                title="Clip Embeddings", 
                num_frames_to_plot=-10, frames=clip_frames[0] if len(clip_frames) > 0 else None)

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
