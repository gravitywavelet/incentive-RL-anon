import os
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


# =========== 1. KL-Reward Wrapper (returns both rewards) ==============
class KLRewardWrapper(gym.Wrapper):
    def __init__(self, env, beta=0.1):
        super().__init__(env)
        self.beta = beta

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_reward = reward  # record before shaping!
        pos = getattr(self.env, 'agent_pos', None)
        goal_pos = (self.env.unwrapped.grid.width - 2, self.env.unwrapped.grid.height - 2)
        if pos is not None and goal_pos is not None:
            dist = np.linalg.norm(np.array(pos) - np.array(goal_pos))
            p_goal = np.exp(-dist) / (np.exp(-dist) + 1)
            predicted = np.array([p_goal, 1 - p_goal])
        else:
            predicted = np.array([0.5, 0.5])
        preferred = np.array([0.999, 0.001])
        eps = 1e-8
        kl = np.sum(predicted * np.log((predicted + eps) / (preferred + eps)))
        shaped_reward = reward - self.beta * kl
        info["env_reward"] = env_reward  # Add original env reward
        info["shaped_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info

# ============ 2. Environment Setup =============
def make_env(beta=0.1, seed=42):
    def thunk():
        env = gym.make("MiniGrid-DoorKey-8x8-v0")
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        env = KLRewardWrapper(env, beta=beta)
        env.reset(seed=seed)
        return env
    return thunk

def make_eval_env(beta=0.1, seed=142):
    def thunk():
        env = gym.make("MiniGrid-DoorKey-8x8-v0")
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        env = KLRewardWrapper(env, beta=beta)
        env.reset(seed=seed)
        return env
    return thunk

# ============ 3. Configurations ================
beta_list = [0.0, 0.01, 0.02, 0.1]   # Multi-beta support
seed_list = [42, 43, 44]       # Multi-seed support
total_timesteps = 2_000_000
eval_freq = 50_000
episodes = 200
n_envs = 32

def train_and_eval(beta, seed):
    print(f"[START] train_and_eval called: beta={beta}, seed={seed}, pid={os.getpid()}", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    env = DummyVecEnv([make_env(beta, seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_eval_env(beta=beta, seed=seed + 200)])
    eval_env = VecTransposeImage(eval_env)

    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=5e-5,
        clip_range=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        device = "cuda"
    )
    #print("Using device:", model.device, flush=True)

    eval_results = []
    timesteps = 0
    while timesteps < total_timesteps:
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        timesteps += eval_freq

        # Evaluation loop
        success_count = 0
        ep_env_rewards, ep_shaped_rewards, ep_lengths = [], [], []
        for _ in range(episodes):
            obs = eval_env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            done, total_env_reward, total_shaped_reward, ep_len = False, 0, 0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = eval_env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result
                obs = next_obs
                if isinstance(info, list):
                    info = info[0]
                env_r = info.get("env_reward", reward)
                shaped_r = info.get("shaped_reward", reward)
                total_env_reward += env_r
                total_shaped_reward += shaped_r
                ep_len += 1
                if env_r > 0:
                    success_count += 1
                    break
            ep_env_rewards.append(total_env_reward)
            ep_shaped_rewards.append(total_shaped_reward)
            ep_lengths.append(ep_len)
        success_rate = success_count / episodes * 100
        mean_env_reward = np.mean(ep_env_rewards)
        mean_total_reward = np.mean(ep_shaped_rewards)
        mean_ep_length = np.mean(ep_lengths)
        eval_results.append({
            "beta": beta,
            "seed": seed,
            "timesteps": timesteps,
            "success_rate": success_rate,
            "mean_env_reward": mean_env_reward,      # before KL shaping
            "mean_total_reward": mean_total_reward,  # after KL shaping
            "mean_length": mean_ep_length
        })
        print(f"[β={beta} | seed={seed} | step {timesteps}] Success: {success_rate:.1f}% | EnvReward: {mean_env_reward:.2f} | ShapedReward: {mean_total_reward:.2f} | Length: {mean_ep_length:.1f}")
    return eval_results

# ============ 4. Multi-seed/Beta Loop ============
print("Starting main loop...", flush=True)
all_results = []
for beta in beta_list:
    for seed in seed_list:
        eval_results = train_and_eval(beta, seed)
        all_results.extend(eval_results)

# ============ 5. Save all results ===============
results_df = pd.DataFrame(all_results)
results_df.to_csv("all_eval_metrics.csv", index=False)
print("✅ All evaluation metrics saved to all_eval_metrics.csv")