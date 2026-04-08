"""
DQN + Transformer Agent cho bài toán MKP (1 ba lô, chọn tuần tự)

Ý tưởng:
    DQN thuần chỉ nhìn vào state hiện tại (vật đang xét).
    Transformer nhìn vào TOÀN BỘ sequence các vật đã qua → học được
    mối quan hệ giữa các vật, quyết định thông minh hơn.

    Ví dụ: "vật A nặng nhưng giá trị cao, còn vật B nhẹ và vừa →
             nên bỏ A để lấy B và C" — Transformer học được điều này.

Kiến trúc:
    [obs_0, obs_1, ..., obs_t]   ← sequence các state đã qua
           ↓
    Linear Embedding             ← chiếu obs_size → d_model
           ↓
    Positional Encoding          ← thêm thông tin thứ tự
           ↓
    TransformerEncoder (L lớp)   ← học quan hệ giữa các vật
           ↓
    Lấy token cuối cùng [t]      ← đại diện cho bước hiện tại
           ↓
    Linear → Q-values [2]        ← Q(s,bỏ) và Q(s,chọn)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

# 1. Positional Encoding

class PositionalEncoding(nn.Module):
    """
    Thêm thông tin vị trí vào sequence
    Vì Transformer không có khái niệm "trước/sau" tự nhiên

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. Transformer Q-Network

class TransformerQNetwork(nn.Module):
    """
    Q-Network dùng Transformer để xử lý sequence các state

    Parameters
    ----------
    obs_size    : kích thước 1 state (= 2k+1)
    d_model     : chiều embedding trong Transformer
    nhead       : số attention head
    num_layers  : số lớp TransformerEncoder
    dropout     : dropout rate
    """
    def __init__(self, obs_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # d_model phải chia hết cho nhead
        assert d_model % nhead == 0, "d_model phải chia hết cho nhead"

        # Chiếu obs_size → d_model
        self.input_proj = nn.Linear(obs_size, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,    # (batch, seq, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: lấy token cuối → Q-values
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),   # Q(bỏ), Q(chọn)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, obs_size)  — sequence các state đã qua
        returns: (batch, 2)            — Q-values tại bước cuối
        """
        # Embed + positional encoding
        x = self.input_proj(x)    # (B, seq, d_model)
        x = self.pos_enc(x)       # (B, seq, d_model)

        # Transformer (tự attention toàn sequence)
        x = self.transformer(x)   # (B, seq, d_model)

        # Lấy token cuối cùng (bước hiện tại)
        x = x[:, -1, :]           # (B, d_model)

        return self.output_head(x)  # (B, 2)

# 3. Sequence Replay Buffer

class SequenceReplayBuffer:
    """
    Buffer lưu trữ TOÀN BỘ sequence của mỗi episode
    Mỗi experience gồm:
        - state_seq     : list các state từ đầu đến bước hiện tại
        - action        : action tại bước hiện tại
        - reward        : reward nhận được
        - next_state_seq: list các state đến bước tiếp theo
        - done          : episode kết thúc chưa
    """
    def __init__(self, capacity=5_000, max_seq_len=200):
        self.buffer      = deque(maxlen=capacity)
        self.max_seq_len = max_seq_len

    def push(self, state_seq, action, reward, next_state_seq, done):
        self.buffer.append((
            list(state_seq),
            action,
            reward,
            list(next_state_seq),
            done
        ))

    def _pad_seq(self, seq, obs_size):
        """Pad sequence về max_seq_len để batch được"""
        seq = seq[-self.max_seq_len:]   # truncate nếu quá dài
        pad_len = self.max_seq_len - len(seq)
        if pad_len > 0:
            pad = [np.zeros(obs_size, dtype=np.float32)] * pad_len
            seq = pad + seq   # pad ở đầu
        return np.array(seq, dtype=np.float32)

    def sample(self, batch_size, obs_size):
        batch = random.sample(self.buffer, batch_size)
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)

        return (
            torch.FloatTensor(np.stack([self._pad_seq(s, obs_size) for s in state_seqs])),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.stack([self._pad_seq(s, obs_size) for s in next_state_seqs])),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

# 4. DQN + Transformer Agent

class DQNTransformerAgent:
    """
    DQN Agent dùng Transformer Q-Network
    Khác DQN thuần: agent nhớ toàn bộ sequence các state đã qua
    trong episode → Transformer học thứ tự chọn vật tốt hơn

    Parameters
    ----------
    obs_size        : kích thước 1 state vector
    d_model         : chiều Transformer embedding
    nhead           : số attention head
    num_layers      : số lớp Transformer
    lr              : learning rate
    gamma           : discount factor
    epsilon_start/end/decay : epsilon-greedy params
    buffer_capacity : dung lượng replay buffer
    batch_size      : số sample mỗi lần train
    target_update   : chu kỳ cập nhật target network
    max_seq_len     : độ dài tối đa của sequence
    """
    def __init__(
        self,
        obs_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=5_000,
        batch_size=32,
        target_update=10,
        max_seq_len=200,
    ):
        self.obs_size    = obs_size
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update = target_update
        self.episode_count = 0
        self.max_seq_len = max_seq_len

        # State sequence hiện tại trong episode
        self.current_seq = []

        # Q-network chính
        self.q_net = TransformerQNetwork(obs_size, d_model, nhead, num_layers)
        # Target network
        self.target_net = TransformerQNetwork(obs_size, d_model, nhead, num_layers)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = SequenceReplayBuffer(buffer_capacity, max_seq_len)
        self.loss_fn   = nn.MSELoss()

    def reset_sequence(self):
        """Gọi khi bắt đầu episode mới"""
        self.current_seq = []

    def _get_padded_seq(self, seq):
        """Pad sequence hiện tại để đưa vào Transformer"""
        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        if pad_len > 0:
            pad = [np.zeros(self.obs_size, dtype=np.float32)] * pad_len
            seq = pad + list(seq)
        return np.array(seq, dtype=np.float32)

    # ── Chọn action ──────────────────────────────────────
    def select_action(self, state, feasible_actions):
        """
        Epsilon-Greedy + Action Masking
        Dùng toàn bộ sequence (không chỉ state hiện tại)
        """
        # Thêm state hiện tại vào sequence
        self.current_seq.append(state)

        if random.random() < self.epsilon:
            return random.choice(feasible_actions)

        seq = self._get_padded_seq(self.current_seq)
        seq_t = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, obs_size)

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(seq_t).squeeze(0)   # (2,)
        self.q_net.train()

        # Action masking
        mask = torch.full((2,), float('-inf'))
        for a in feasible_actions:
            mask[a] = q_values[a]

        return mask.argmax().item()

    # ── Lưu experience ───────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        """
        Lưu sequence hiện tại vào buffer
        next_state_seq = current_seq + [next_state]
        """
        state_seq      = list(self.current_seq)
        next_state_seq = state_seq + [next_state]
        self.buffer.push(state_seq, action, reward, next_state_seq, done)

    # ── Train 1 bước ─────────────────────────────────────
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        state_seqs, actions, rewards, next_state_seqs, dones = \
            self.buffer.sample(self.batch_size, self.obs_size)

        # Q(seq, a)
        q_values = self.q_net(state_seqs)                          # (B, 2)
        q_pred   = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # TD target
        with torch.no_grad():
            q_next   = self.target_net(next_state_seqs).max(1)[0]  # (B,)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── Kết thúc episode ─────────────────────────────────
    def end_episode(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.reset_sequence()

        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())