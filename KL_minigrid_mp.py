import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, TimeoutError
import multiprocessing as mp
import random


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
beta_list = [0.0, 0.005, 0.01, 0.02, 0.1]   # Multi-beta support
seed_list = [42, 43, 44, 45, 46]       # Multi-seed support
total_timesteps = 3_000_000
eval_freq = 50_000
episodes = 200
n_envs = 16

def train_and_eval(beta, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logdir = f"/root/tf-logs/doorkey_beta{beta}_seed{seed}"
    writer = SummaryWriter(logdir)

    env = DummyVecEnv([make_env(beta, seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_eval_env(beta=beta, seed=seed + 200)])
    eval_env = VecTransposeImage(eval_env)

    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        n_steps=128,
        batch_size=512,
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
        writer.add_scalar("success_rate", success_rate, timesteps)
        writer.add_scalar("mean_env_reward", mean_env_reward, timesteps)
        writer.add_scalar("mean_shaped_reward", mean_total_reward, timesteps)
        writer.add_scalar("mean_length", mean_ep_length, timesteps)

        print(f"[β={beta} | seed={seed} | step {timesteps}] Success: {success_rate:.1f}% | EnvReward: {mean_env_reward:.2f} | ShapedReward: {mean_total_reward:.2f} | Length: {mean_ep_length:.1f}")
    model_save_path = f"saved/ppo_doorkey_beta{beta}_seed{seed}_final.zip"
    model.save(model_save_path)
    return eval_results

    
    
def run_and_save(beta, seed):
    import traceback
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[WORKER STARTED] beta={beta}, seed={seed}, pid={os.getpid()}", flush=True)
        results = train_and_eval(beta, seed)
        out_path = f"results/eval_beta{beta}_seed{seed}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"✅ beta={beta}, seed={seed} saved!")
    except Exception as e:
        # 详细记录异常栈
        print(f"❌ [ERROR] beta={beta}, seed={seed}, pid={os.getpid()}\nException: {e}", flush=True)
        traceback.print_exc()
        # 可以考虑将失败信息单独保存（比如写个空文件或异常日志）
        fail_log = f"results/FAILED_beta{beta}_seed{seed}.log"
        with open(fail_log, "w") as f:
            f.write(f"beta={beta}, seed={seed}\nException: {e}\n")
            f.write(traceback.format_exc())

    
if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tasks = [(beta, seed) for beta in beta_list for seed in seed_list]
    with mp.Pool(processes=10) as pool:
        results = [pool.apply_async(run_and_save, t) for t in tasks]
        for r in results:
            try:
                r.get(timeout=12*3600)  # 最多等待12小时/任务
            except Exception as e:
                print("A worker crashed or was killed:", e)

    dfs = [pd.read_csv(f) for f in glob.glob("results/eval_beta*_seed*.csv")]
    pd.concat(dfs).to_csv("results/all_eval_metrics.csv", index=False)
    print("✅ All evaluation metrics saved to all_eval_metrics.csv")