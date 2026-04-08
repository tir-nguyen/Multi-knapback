"""
DQN Agent cho bài toán MKP (1 ba lô, chọn tuần tự)

MDP theo đề tài:
    State  : [dung_luong_con_lai (k chiều, chuẩn hóa)
              + value vật hiện tại (chuẩn hóa)
              + weight vật hiện tại theo k chiều (chuẩn hóa)]
    Action : 0 = bỏ vật, 1 = chọn vật
    Reward : giá trị vật nếu chọn hợp lệ, 0 nếu bỏ hoặc vi phạm

Kỹ thuật DQN:
    - Experience Replay   : lưu (s, a, r, s', done) → train từ random mini-batch
    - Target Network      : mạng riêng tính TD target, cập nhật chậm
    - Epsilon-Greedy      : cân bằng exploration vs exploitation
    - Action Masking      : chặn action vi phạm ràng buộc ngay từ đầu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 1. Q-Network  (MLP thuần, không Transformer)

class QNetwork(nn.Module):
    """
    Mạng neural xấp xỉ Q(s, a)
    Input  : state vector  (obs_size,)
    Output : Q-value cho 2 action [Q(s,bỏ), Q(s,chọn)]
    """
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),   # 2 action: bỏ(0) hoặc chọn(1)
        )

    def forward(self, x):
        return self.net(x)

# 2. Replay Buffer  (Experience Replay)

class ReplayBuffer:
    """
    Lưu trữ experience (s, a, r, s', done)
    Khi đầy → tự xóa experience cũ nhất (deque)
    """
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

# 3. DQN Agent

class DQNAgent:
    """
    DQN Agent cho MKP 1 ba lô, chọn tuần tự

    Parameters
    ----------
    obs_size        : kích thước state vector
    hidden_size     : số neuron mỗi lớp ẩn
    lr              : learning rate
    gamma           : discount factor (coi trọng reward tương lai)
    epsilon_start   : epsilon ban đầu (exploration cao)
    epsilon_end     : epsilon tối thiểu
    epsilon_decay   : tốc độ giảm epsilon mỗi episode
    buffer_capacity : dung lượng replay buffer
    batch_size      : số sample mỗi lần train
    target_update   : cập nhật target network mỗi bao nhiêu episode
    """
    def __init__(
        self,
        obs_size,
        hidden_size=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10_000,
        batch_size=64,
        target_update=10,
    ):
        self.obs_size    = obs_size
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update = target_update
        self.episode_count = 0

        # Q-network chính (train liên tục)
        self.q_net    = QNetwork(obs_size, hidden_size)
        # Target network (cập nhật chậm → ổn định training)
        self.target_net = QNetwork(obs_size, hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self.loss_fn   = nn.MSELoss()

    # ── Chọn action ──────────────────────────────────────
    def select_action(self, state, feasible_actions):
        """
        Epsilon-Greedy + Action Masking:
            - Với xác suất epsilon → chọn ngẫu nhiên trong feasible_actions
            - Còn lại → chọn action có Q cao nhất (trong feasible_actions)

        Action Masking: nếu chọn vật sẽ vi phạm ràng buộc
        thì feasible_actions = [0], agent buộc phải bỏ vật đó.
        """
        if random.random() < self.epsilon:
            return random.choice(feasible_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, obs_size)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)   # (2,)

        # Mask các action không hợp lệ bằng -inf
        mask = torch.full((2,), float('-inf'))
        for a in feasible_actions:
            mask[a] = q_values[a]

        return mask.argmax().item()

    # ── Lưu experience ───────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Train 1 bước ─────────────────────────────────────
    def train_step(self):
        """
        Lấy mini-batch từ buffer → tính TD target → cập nhật Q-network

        TD target: y = r + γ * max_a Q_target(s', a)   (nếu chưa done)
                   y = r                                (nếu done)
        Loss: MSE( Q(s,a) , y )
        """
        if len(self.buffer) < self.batch_size:
            return None   # chưa đủ data

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Q(s, a) — chỉ lấy Q của action đã thực hiện
        q_values = self.q_net(states)                          # (B, 2)
        q_pred   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # TD target: r + γ * max Q_target(s')
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(1)[0]  # (B,)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── Kết thúc episode ─────────────────────────────────
    def end_episode(self):
        """Giảm epsilon + cập nhật target network theo chu kỳ"""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())