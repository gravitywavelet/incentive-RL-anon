import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
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


# TOTAL_TIMESTEPS = 20_000
# EVAL_FREQ       = 10_000
TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ       = 100_000

EVAL_EPISODES   = 100
N_ENVS          = 8
SCORE_THRESH    = 0.1
MIN_BETA = 1e-5


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

# ============ 2b. utility functions =============

def extract_valid_candidates(results):
    """
    Filter (beta, normalized_score) from results with non-zero success rate.

    Args:
        results: list of tuples (seed, round, beta, score)

    Returns:
        List of tuples (beta, score_normalized_to_0_1)
    """
    return [(b, s / 100.0) for (_, _, b, s) in results if s > 0]


def compute_beta_weights_linear(beta_scores):
    """
    Use raw normalized scores as linear weights (no softmax).
    If all scores are zero, returns None, as this functions should not be called in this case.

    Args:
        beta_scores: list of (beta, normalized_score ∈ [0,1])

    Returns:
        list of (beta, weight), or None if empty
    """
    if not beta_scores:
        print("⚠️ No successful beta this round — skip calc weights")
        return None

    scores = np.array([s for _, s in beta_scores])
    betas = np.array([b for b, _ in beta_scores])

    total_score = scores.sum()
    if total_score == 0:
        return None

    weights = scores / total_score
    return list(zip(betas, weights))

def compute_weighted_mean_variance(beta_weight_pairs):
    """
    Computes the weighted mean and variance of β values.

    Args:
        beta_weight_pairs: list of (beta, weight)

    Returns:
        (mean_beta, var_beta)
    """
    betas = np.array([b for b, _ in beta_weight_pairs])
    weights = np.array([w for _, w in beta_weight_pairs])

    mean_beta = np.sum(weights * betas)
    var_beta = np.sum(weights * (betas - mean_beta) ** 2)
    var_beta = max(var_beta, 1e-6)  # numerical safeguard

    return mean_beta, var_beta

def moment_match_deltas(mean_beta, var_beta, alpha_old, beta_old):
    """
    Suggests how to shift (alpha, beta) using moment-matching.

    Args:
        mean_beta: weighted average of good β values
        var_beta: weighted variance
        alpha_old, beta_old: current Beta distribution parameters

    Returns:
        delta_alpha, delta_beta: how much to move (α, β) toward the new distribution
    """
    # Estimate target alpha and beta via moment-matching
    common = (mean_beta * (1 - mean_beta)) / var_beta - 1
    alpha_target = mean_beta * common
    beta_target = (1 - mean_beta) * common

    # Return the deltas (step toward the target)
    delta_alpha = alpha_target - alpha_old
    delta_beta = beta_target - beta_old

    return delta_alpha, delta_beta

def scale_concentration(alpha, beta_param, round_idx,
                        min_conc=2.0, max_conc=100.0, growth=1.618):
    """
    Scales Beta(α, β) concentration to control exploration vs. exploitation.

    Args:
        alpha, beta_param: current Beta distribution parameters
        round_idx: index of current round (0-based)
        min_conc: base concentration (encourages exploration early)
        max_conc: upper bound to avoid overfitting
        growth: geometric rate of concentration tightening (e.g. golden ratio)

    Returns:
        (scaled_alpha, scaled_beta): adjusted parameters with controlled sharpness
    """
    # Compute target concentration for this round
    target_conc = min_conc * (growth ** (round_idx+3))
    target_conc = min(target_conc, max_conc)

    # Current concentration level
    current_conc = alpha + beta_param
    scale = target_conc / current_conc
    print(f"alpha:{alpha}, beta:{beta_param}, scale:{scale}")

    return alpha * scale, beta_param * scale



# ============ 3. Training/Eval =============
def train_and_eval(beta=0.0, seed=42, total_timesteps=TOTAL_TIMESTEPS, eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES, n_envs=N_ENVS, round_idx=1):
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
        
        if timesteps < TOTAL_TIMESTEPS / 2:
            print(f"round:{round_idx+1} | beta:{beta} | Skipping eval at step {timesteps}")
            continue
            
        for _ in range(eval_episodes):
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
        success_rate = success_count / eval_episodes * 100
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

        print(f"[β={beta:.5f} | seed={seed} | step {timesteps}] Success: {success_rate:.1f}% | EnvReward: {mean_env_reward:.2f} | ShapedReward: {mean_total_reward:.2f} | Length: {mean_ep_length:.1f}")
    model_save_path = f"saved/ppo_doorkey_seed{seed}_round_{round_idx+1}_beta{beta:.5f}_sr{int(success_rate)}.zip"
    model.save(model_save_path)
    return eval_results

#def run_and_save(beta, seed=42, round_idx=0):
def run_and_save(beta, seed=42, total_timesteps=2_000_000, eval_freq=50_000, eval_episodes=EVAL_EPISODES, n_envs=8, round_idx=0):
    import traceback
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[WORKER] beta={beta:.5f}, seed={seed}, pid={os.getpid()}", flush=True)
        
        results = train_and_eval(beta=beta, seed=seed, total_timesteps=total_timesteps, eval_freq=eval_freq, eval_episodes=EVAL_EPISODES, n_envs=n_envs, round_idx=round_idx)
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
    return True #as running in parralel
            

def beta_search(
    seeds=[42, 47, 101],
    total_rounds=5,
    total_timesteps=TOTAL_TIMESTEPS,
    eval_freq=EVAL_FREQ,
    eval_episodes=EVAL_EPISODES,
    n_envs=N_ENVS,
    betas_per_round=10
):
    all_history = []
    beta_log_records = []

    alpha, beta_param = 1, 1   # shared posterior across all seeds

    for round_idx in range(total_rounds):
        start_time = datetime.now()
        print(f"\n⏱️ [Round {round_idx+1}] Start at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        rng = np.random.default_rng(999 + round_idx)  # global RNG, independent of seed
        sampled_betas = list(beta_dist(alpha, beta_param).rvs(betas_per_round, random_state=rng))
        if round_idx == total_rounds - 1:
            sampled_betas.append(0.0)  # baseline

        sampled_betas = sorted(round(b, 5) for b in sampled_betas)
        sampled_betas = np.clip(sampled_betas, MIN_BETA, 1.0)
        print(f"--- Round {round_idx+1} sampling: {sampled_betas} ---")

        # run all (seed, beta) pairs
        round_results = []
        for seed in seeds:
            results = []
            with mp.Pool(processes=len(sampled_betas)) as pool:
                rets = [
                    pool.apply_async(
                        run_and_save,
                        (b, seed, total_timesteps, eval_freq, eval_episodes, n_envs, round_idx)
                    )
                    for b in sampled_betas
                ]
                for (b, r) in zip(sampled_betas, rets):
                    try:
                        r.get(timeout=3600*4)
                        matches = list(Path("results").glob(f"eval_seed{seed}_round*_beta{b:.5f}_sr*.csv"))
                        if matches:
                            df = pd.read_csv(matches[0])
                            score = df.iloc[-1].get("success_rate", 0)
                        else:
                            score = 0
                        results.append((seed, round_idx+1, b, score))
                    except Exception as e:
                        print(f"⚠️ Worker crashed for beta={b:.5f}, seed={seed}, err={e}")
                        results.append((seed, round_idx+1, b, 0))
            round_results.extend(results)

        # aggregate across seeds before updating posterior
        history_df = pd.DataFrame(round_results, columns=["seed", "round", "beta", "score"])
        all_history.extend(round_results)

        candidates = extract_valid_candidates(round_results)
        print(f"candidates:{candidates}, count={len(candidates)}")

        beta_weight_pairs = compute_beta_weights_linear(candidates)
        if beta_weight_pairs is None:
            delta_alpha, delta_beta = 0, 10
        else:
            mean_beta, var_beta = compute_weighted_mean_variance(beta_weight_pairs)
            delta_alpha, delta_beta = moment_match_deltas(mean_beta, var_beta, alpha, beta_param)

        alpha += delta_alpha
        beta_param += delta_beta
        alpha, beta_param = scale_concentration(alpha, beta_param, round_idx)

        mean_beta_post = alpha / (alpha + beta_param)
        msg = f"✅ [Round {round_idx+1} End] Updated Beta({alpha:.2f}, {beta_param:.2f}) → mean β ≈ {mean_beta_post:.5f}"
        print(msg)
        logging.info(msg)

        # log summary per round
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "round": round_idx+1,
            "n_betas": len(sampled_betas),
            "alpha_param": alpha,
            "beta_param": beta_param,
            "mean_beta_all": np.mean(sampled_betas),
            "std_beta_all": np.std(sampled_betas),
            "min_beta": min(sampled_betas),
            "max_beta": max(sampled_betas),
            "betas": sampled_betas,
        }
        beta_log_records.append(record)
        csv_path = "results/beta_log_summary_all.csv"
        pd.DataFrame([record]).to_csv(
            csv_path,
            mode="a",
            header=not Path(csv_path).exists(),
            index=False
        )

    pd.DataFrame(all_history, columns=["seed","round","beta","score"]).to_csv(
        "results/beta_search_history_all.csv", index=False
    )
    print("✅ Round-major multi-seed Bayesian beta search finished!")


if __name__ == "__main__":
    
    seeds=[42, 47, 101]
    total_rounds = 5
    betas_per_round = 10
    
    beta_search(
        seeds=seeds,
        total_rounds=total_rounds,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        betas_per_round = betas_per_round,
        n_envs=N_ENVS
    )
