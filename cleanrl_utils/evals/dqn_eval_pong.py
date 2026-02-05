import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

def save_envs(envs):
    return [env.unwrapped.ale.cloneSystemState() for env in envs]

def restore_envs(env_id, saved_env_states, idx, capture_video, run_name):
    return gym.vector.SyncVectorEnv([restore_env(env_id, saved_env_state, idx, capture_video, run_name) for 
            saved_env_state in saved_env_states])

def get_actions(q_values):
    print(q_values)
    test = torch.topk(q_values, k=4)
    print(test)
    print(torch.argmax(q_values, dim=1).cpu().numpy())
    return torch.argmax(q_values, dim=1).cpu().numpy()




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
    i = 0
    while len(episodic_returns) < eval_episodes:
        i += 1
        # if random.random() < epsilon:
        #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # else:
        q_values = model(torch.Tensor(obs).to(device))
        actions = get_actions(q_values)
        if i == 10:
            return
        # print(actions)
        cnn_fts = activation['cnn_features']
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
