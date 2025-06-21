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
import multiprocessing as mp
import random
from scipy.stats import beta as beta_dist
from datetime import datetime
from pathlib import Path
import csv
import logging

logging.basicConfig(
    filename='beta_sampling.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# =========== 1. KL-Reward Wrapper ==============
class KLRewardWrapper(gym.Wrapper):
    def __init__(self, env, beta=0.1):
        super().__init__(env)
        self.beta = beta

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_reward = reward
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
        info["env_reward"] = env_reward
        info["shaped_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info

# ============ 2. Env Setup =============
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


# ============ 3. Training/Eval =============
#def train_and_eval(beta, seed=42, total_timesteps=10_000, eval_freq=5_000, episodes=50, n_envs=8, writer=None):
def train_and_eval(beta, seed=42, total_timesteps=2_000_000, eval_freq=50_000, episodes=50, n_envs=8, round_idx=1, writer=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    env = DummyVecEnv([make_env(beta, seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_eval_env(beta=beta, seed=seed + 200)])
    eval_env = VecTransposeImage(eval_env)

    model = PPO(
        'CnnPolicy',
        env,
        verbose=0,
        n_steps=128,
        batch_size=512,
        learning_rate=5e-5,
        clip_range=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        device="cuda"
    )

    eval_results = []
    timesteps = 0
    while timesteps < total_timesteps:
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        timesteps += eval_freq

        # Evaluation
        success_count, ep_env_rewards, ep_shaped_rewards, ep_lengths = 0, [], [], []
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
                if isinstance(info, list): info = info[0]
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
            "mean_env_reward": mean_env_reward,
            "mean_total_reward": mean_total_reward,
            "mean_length": mean_ep_length
        })
        if writer is not None:
            tag_prefix = f"Seed{seed}/Beta{beta:.5f}"
            writer.add_scalar(f"{tag_prefix}/success_rate", success_rate, timesteps)
            writer.add_scalar(f"{tag_prefix}/mean_env_reward", mean_env_reward, timesteps)
            writer.add_scalar(f"{tag_prefix}/mean_shaped_reward", mean_total_reward, timesteps)
            writer.add_scalar(f"{tag_prefix}/mean_length", mean_ep_length, timesteps)


        print(f"[β={beta:.5f} | seed={seed} | step {timesteps}] Success: {success_rate:.1f}% | EnvReward: {mean_env_reward:.2f} | ShapedReward: {mean_total_reward:.2f} | Length: {mean_ep_length:.1f}")
    model_save_path = f"saved/ppo_doorkey_seed{seed}_round_{round_idx}_beta{beta:.5f}_sr{int(success_rate)}.zip"
    model.save(model_save_path)
    return eval_results

def run_and_save(beta, seed=42, round_idx=0, writer=None):
    import traceback
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[WORKER] beta={beta:.5f}, seed={seed}, pid={os.getpid()}", flush=True)
        results = train_and_eval(beta, seed, round_idx, writer=writer)
        score = results[-1]["success_rate"]
        score_percent = score
        out_path = f"results/eval_seed{seed}_round{round_idx+1}_beta{beta:.5f}_sr{score_percent:.1f}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"✅ seed={seed} round={round_idx+1} beta={beta:.5f} saved!")
    except Exception as e:
        print(f"❌ [ERROR] beta={beta:.5f}, seed={seed}, pid={os.getpid()}\nException: {e}", flush=True)
        traceback.print_exc()
        fail_log = f"results/FAILED_beta{beta:.5f}_seed{seed}.log"
        with open(fail_log, "w") as f:
            f.write(f"beta={beta}, seed={seed} round={round_idx+1} \nException: {e}\n")
            import traceback as tb
            f.write(tb.format_exc())



def beta_search(total_rounds=4, betas_per_round=10, seeds=[42,43], writer=None):
    all_history = []
    beta_log_records = []

    for seed in seeds:
        print(f"\n🎯 Running for seed {seed}")
        alpha , beta_param = 1, 1
        history = []

        for round_idx in range(total_rounds):
            start_time = datetime.now()
            print(f"\n⏱️ [Seed {seed} | Round {round_idx+1}] Start at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
           
            rng = np.random.default_rng(seed * 100 + round_idx)  # 防冲突，可调
            sampled_betas = list(beta_dist(alpha , beta_param).rvs(betas_per_round, random_state=rng))
            #sampled_betas = list(beta_dist(alpha, beta_param).rvs(betas_per_round, random_state=rng) * 0.2)
            if round_idx == total_rounds - 1:
                sampled_betas.append(0.0)  # baseline
                
            sampled_betas = sorted(set(round(b, 5) for b in sampled_betas))  # keep precision to 5 digits
           
            
            mean_beta = np.mean(sampled_betas) # all Betas
            std_beta = np.std(sampled_betas)
            
            msg = (f"--- Round {round_idx+1} sampling: {sampled_betas} "
                   f"(n={len(sampled_betas)}, mean={mean_beta:.5f}, std={std_beta:.5f}) ---")
            print(msg)
            logging.info(msg)
            

            results = []
            with mp.Pool(processes=len(sampled_betas)) as pool:

                rets = [pool.apply_async(run_and_save, (b, seed, round_idx)) for b in sampled_betas]
                for (b, r) in zip(sampled_betas, rets):
                    try:
                        r.get(timeout=3600*4)
                        
                        matches = list(Path("results").glob(f"eval_seed{seed}_round*_beta{b:.5f}_sr*.csv"))
                        if matches:
                            df = pd.read_csv(matches[0])
                            score = df.iloc[-1].get("success_rate", 0)
                        else:
                            print(f"❗ No result file found for beta={b:.5f}, seed={seed}")
                            score = 0
                        results.append((seed, round_idx + 1, b, score))   # (42,1,0.012,86.0)
 

                    except Exception as e:
                        error_msg = f"⚠️ A worker crashed for beta={b:.5f}, error: {str(e)}"
                        print(error_msg)
                        results.append((seed, round_idx + 1, b, 0))

                        crash_tag = f"Seed{seed}/Round{round_idx+1}/crash_beta_{b:.5f}"
                        if writer:
                            writer.add_text(crash_tag, error_msg, round_idx)
            
            history.extend(results)
            

            # Update Beta distribution
            df_hist = pd.DataFrame(results, columns=["seed", "round", "beta", "score"])
            print(f"df_hist:{df_hist}")
            
            non_zero_rows = df_hist[df_hist["score"] > 1e-6]
            
            scores = non_zero_rows["score"].to_numpy()
            betas = non_zero_rows["beta"].to_numpy()
            non_zero_scores_count = len(scores) 
            zero_scores_count = len(df_hist) - non_zero_scores_count 
            
            if non_zero_scores_count > 0:
                scores_norm = np.clip(scores / 100.0, 0, 1)  # Normalize to [0, 1]
                
                #alpha_update = scores_norm.sum()
                beta_update = (1 - scores_norm).sum() + zero_scores_count

                #alpha += alpha_update
                beta_param += beta_update

                mean_beta = alpha / (alpha + beta_param)

                msg = f"✅ [Round {round_idx+1}] Updated Beta({alpha:.2f}, {beta_param:.2f}) → mean β ≈ {mean_beta:.5f} | non-zero beta count={non_zero_scores_count}"
                print(msg)
                logging.info(msg)
            else:
                beta_param += 1.0 
                print(f"⚠️  [Round {round_idx+1}] All βs failed → shift posterior toward smaller β")
                #mean_beta = 0.5 # fallback if no signal
                #print(f"⚠️ [Round {round_idx+1}] All scores are zero, fallback to unweighted mean_beta: {mean_beta:.5f}")
                
            

            # 日志记录：每轮 beta 分布采样情况
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed,
                "round": round_idx + 1,
                "n_betas": len(sampled_betas),
                "alpha":alpha,
                "beta_param":beta_param,
                "mean_beta_all": np.mean(sampled_betas),
                "std_beta_all": np.std(sampled_betas),
                "min_beta": min(sampled_betas),
                "max_beta": max(sampled_betas),
                "betas": sampled_betas,
            }
            beta_log_records.append(record)

            # 写入统一日志文件 per round
            csv_path = "results/beta_log_summary_all.csv"
            df_curr = pd.DataFrame([record])
            df_curr.to_csv(
                csv_path,
                mode="a",  # 追加模式
                header=not Path(csv_path).exists(),  # 如果文件不存在则写 header
                index=False
            )

        all_history.extend(history)

    pd.DataFrame(all_history).to_csv("results/beta_search_history_all.csv", index=False)
    print("✅ Multi-seed Bayesian beta search finished!")


if __name__ == "__main__":
    
    total_rounds = 4
    seeds=[42]

    mp.set_start_method("spawn", force=True)
    #ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_logdir = f"/root/tf-logs/"
 
    writer = SummaryWriter(root_logdir)
    beta_search(writer=writer, total_rounds=total_rounds, seeds=seeds)
    writer.close()