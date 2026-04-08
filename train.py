"""
Training & Evaluation Script
So sánh: DQN | DQN+Transformer | Greedy

Chạy:
    python train.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mkp_env import make_random_mkp
from dqn_agent import DQNAgent
from dqn_transformer_agent import DQNTransformerAgent

# Greedy Baseline

def greedy_solve(env):
    """
    Greedy chuẩn cho MKP 1 ba lô:
        1. Tính ratio = value / tổng_weight cho mỗi vật
        2. Sắp xếp vật theo ratio giảm dần → đây là thứ tự ưu tiên
        3. Xét tuần tự từng vật theo env (index 0→n-1):
           - Nếu vật hiện tại nằm trong top priority VÀ còn vừa ba lô → chọn
           - Ngược lại → bỏ
    Không cần cắt top n/2, tự nhiên sẽ dừng khi ba lô đầy
    """
    obs = env.reset()
    total_value = 0.0
    selected = []

    # Tính ratio giá trị / tổng trọng lượng (chuẩn Greedy MKP)
    ratios = env.values / (env.weights.sum(axis=1) + 1e-8)
    # priority_set: tập vật được ưu tiên chọn (theo ratio)
    # Sắp xếp giảm dần → vật nào ratio cao hơn thì ưu tiên hơn
    priority_rank = {item: rank for rank, item in enumerate(np.argsort(-ratios))}

    # Ngưỡng: ưu tiên chọn vật có rank cao (top 70%)
    threshold = int(env.n_items * 0.7)

    for i in range(env.n_items):
        feasible = env.get_feasible_actions()
        # Chọn nếu: vật nằm trong top priority VÀ còn vừa ba lô (action 1 hợp lệ)
        want_pick = priority_rank[i] < threshold
        action = 1 if want_pick and (1 in feasible) else 0
        obs, reward, done, info = env.step(action)
        total_value += reward
        if info['valid_pick']:
            selected.append(i)

    return total_value, selected


# Training loop chung

def train(agent, n_items, n_constraints, n_episodes=500, seed=None, verbose=True):
    """
    Huấn luyện agent trên nhiều episode
    Mỗi episode là 1 bài MKP ngẫu nhiên mới (để agent học tổng quát)

    Returns: list reward trung bình mỗi 50 episode
    """
    rewards_history = []
    running_reward  = 0.0
    is_transformer  = hasattr(agent, 'reset_sequence')

    for ep in range(1, n_episodes + 1):
        # Tạo bài toán mới mỗi episode
        env = make_random_mkp(n_items, n_constraints,
                              seed=None if seed is None else seed + ep)
        obs  = env.reset()
        done = False
        ep_reward = 0.0

        if is_transformer:
            agent.reset_sequence()

        while not done:
            feasible = env.get_feasible_actions()
            action   = agent.select_action(obs, feasible)
            next_obs, reward, done, info = env.step(action)

            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()

            obs = next_obs
            ep_reward += reward

        agent.end_episode()
        running_reward += ep_reward

        if ep % 50 == 0:
            avg = running_reward / 50
            rewards_history.append(avg)
            running_reward = 0.0
            if verbose:
                print(f"  Episode {ep:4d} | avg_reward={avg:.2f} | epsilon={agent.epsilon:.3f}")

    return rewards_history


# Evaluation (inference không explore)


def evaluate(agent, n_items, n_constraints, n_eval=20, seed=100):
    """Đánh giá agent sau khi train (epsilon = 0)"""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    is_transformer = hasattr(agent, 'reset_sequence')

    total_rewards = []
    for i in range(n_eval):
        env = make_random_mkp(n_items, n_constraints, seed=seed + i)
        obs  = env.reset()
        done = False
        ep_reward = 0.0

        if is_transformer:
            agent.reset_sequence()

        while not done:
            feasible = env.get_feasible_actions()
            action   = agent.select_action(obs, feasible)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)

    agent.epsilon = old_eps
    return np.mean(total_rewards), np.std(total_rewards)


if __name__ == "__main__":
    N_ITEMS       = 10
    N_CONSTRAINTS = 3
    N_EPISODES    = 300
    SEED          = 42

    # Kích thước state = 2*k + 1
    OBS_SIZE = 2 * N_CONSTRAINTS + 1

    print("=" * 55)
    print("   MKP - Học tăng cường (1 ba lô, chọn tuần tự)")
    print(f"   Vật phẩm: {N_ITEMS} | Ràng buộc: {N_CONSTRAINTS} chiều")
    print("=" * 55)

    # ── Greedy baseline ──────────────────────────────────
    print("\n[Greedy Baseline]")
    greedy_rewards = []
    for i in range(20):
        env = make_random_mkp(N_ITEMS, N_CONSTRAINTS, seed=100 + i)
        val, _ = greedy_solve(env)
        greedy_rewards.append(val)
    greedy_mean = np.mean(greedy_rewards)
    greedy_std  = np.std(greedy_rewards)
    print(f"  Reward trung bình: {greedy_mean:.2f} ± {greedy_std:.2f}")

    # ── DQN ─────────────────────────────────────────────
    print(f"\n[DQN] Training {N_EPISODES} episodes...")
    dqn_agent = DQNAgent(
        obs_size=OBS_SIZE,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        batch_size=64,
        target_update=10,
    )
    train(dqn_agent, N_ITEMS, N_CONSTRAINTS, N_EPISODES, seed=SEED)
    dqn_mean, dqn_std = evaluate(dqn_agent, N_ITEMS, N_CONSTRAINTS, seed=100)
    print(f"  Reward trung bình: {dqn_mean:.2f} ± {dqn_std:.2f}")

    # ── DQN + Transformer ────────────────────────────────
    print(f"\n[DQN + Transformer] Training {N_EPISODES} episodes...")
    dqn_tf_agent = DQNTransformerAgent(
        obs_size=OBS_SIZE,
        d_model=64,
        nhead=4,
        num_layers=2,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        batch_size=32,
        target_update=10,
        max_seq_len=N_ITEMS + 1,
    )
    train(dqn_tf_agent, N_ITEMS, N_CONSTRAINTS, N_EPISODES, seed=SEED)
    tf_mean, tf_std = evaluate(dqn_tf_agent, N_ITEMS, N_CONSTRAINTS, seed=100)
    print(f"  Reward trung bình: {tf_mean:.2f} ± {tf_std:.2f}")

    # ── Tổng kết ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   KẾT QUẢ SO SÁNH")
    print("=" * 55)
    print(f"  {'Phương pháp':<22} {'Reward TB':>10} {'Std':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Greedy':<22} {greedy_mean:>10.2f} {greedy_std:>8.2f}")
    print(f"  {'DQN':<22} {dqn_mean:>10.2f} {dqn_std:>8.2f}")
    print(f"  {'DQN + Transformer':<22} {tf_mean:>10.2f} {tf_std:>8.2f}")
    print("=" * 55)