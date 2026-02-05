import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import statistics


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
        print(f"Cosine similarity between feature at index {idx} and index {idx+1}: {sim}")
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
        print(f"Cosine similarity between feature at index {idx1} and index {idx2}: {sim}")
    print(f"Average cosine similarity between random frames: {statistics.mean(random_vals)}")



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
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.network[8].register_forward_hook(get_activation('cnn_features'))

    obs, _ = envs.reset()
    episodic_returns = []
    cnn_list = []

    i = 0
    while len(episodic_returns) < eval_episodes:
        i += 1
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values, best_action_prob=1.0)
        cnn_fts = activation['cnn_features']
        if i > 500 and i < 1000:
            cnn_list.append(cnn_fts.detach().cpu().clone())
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        # if i == 1000:
        #     print("saving and reloading envs")
        #     saved_envs = save_envs(envs.envs)
        #     envs.close()
        #     envs = restore_envs(env_id, saved_envs, 0, capture_video, run_name)
        obs = next_obs
    
    cnn_similarity(cnn_list)
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
    )
