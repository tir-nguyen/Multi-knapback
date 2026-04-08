"""
MKP Environment - Multi-dimensional Knapsack Problem
Gym-style environment cho bài toán Ba lô nhiều chiều

Bài toán:
    - n vật phẩm, mỗi vật có: giá trị v_i và k ràng buộc w_ij
    - Ba lô có k giới hạn c_j
    - Chọn vật sao cho tổng giá trị lớn nhất, không vượt quá giới hạn

MDP:
    - State : [dung_luong_con_lai (k chiều) + thông tin vật hiện tại (k+1)]
    - Action: 0 = bỏ vật, 1 = chọn vật
    - Reward: giá trị vật nếu chọn hợp lệ, 0 nếu bỏ hoặc vi phạm
"""

import numpy as np

class MKPEnv:
    """
    Multi-dimensional Knapsack Problem Environment (Gym-style)

    Parameters
    ----------
    n_items       : số lượng vật phẩm
    n_constraints : số ràng buộc k (số chiều của ba lô)
    capacity      : giới hạn của ba lô, shape (k,)
    values        : giá trị của từng vật, shape (n,)
    weights       : trọng lượng theo từng chiều, shape (n, k)
    """

    def __init__(self, n_items, n_constraints, capacity, values, weights):
        self.n_items = n_items
        self.n_constraints = n_constraints
        self.capacity = np.array(capacity, dtype=np.float32)
        self.values = np.array(values, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)  # shape (n, k)

        # Kích thước state vector:
        # [dung_luong_con_lai * k] + [value_vat_hien_tai] + [weights_vat_hien_tai * k]
        self.obs_size = n_constraints + 1 + n_constraints  # = 2k + 1

        self.reset()

    def reset(self):
        """Reset môi trường về đầu episode"""
        self.current_item = 0                              # đang xét vật thứ mấy
        self.remaining_capacity = self.capacity.copy()    # dung lượng còn lại
        self.selected = []                                 # danh sách vật đã chọn
        self.total_value = 0.0                             # tổng giá trị hiện tại
        return self._get_obs()

    def _get_obs(self):
        """
        Tạo vector observation (state) cho agent

        Gồm:
            - dung lượng còn lại (chuẩn hóa về [0,1])
            - giá trị vật hiện tại (chuẩn hóa)
            - trọng lượng vật hiện tại theo từng chiều (chuẩn hóa)
        """
        # Chuẩn hóa dung lượng còn lại
        norm_capacity = self.remaining_capacity / self.capacity

        if self.current_item < self.n_items:
            # Chuẩn hóa giá trị vật hiện tại
            norm_value = np.array(
                [self.values[self.current_item] / (self.values.max() + 1e-8)],
                dtype=np.float32
            )
            # Chuẩn hóa trọng lượng vật hiện tại theo từng chiều
            norm_weight = (
                self.weights[self.current_item] /
                (self.capacity + 1e-8)
            )
        else:
            # Đã hết vật → padding bằng 0
            norm_value = np.zeros(1, dtype=np.float32)
            norm_weight = np.zeros(self.n_constraints, dtype=np.float32)

        obs = np.concatenate([norm_capacity, norm_value, norm_weight])
        return obs.astype(np.float32)

    def step(self, action):
        """
        Thực hiện action tại bước hiện tại

        Parameters
        ----------
        action : 0 (bỏ vật) hoặc 1 (chọn vật)

        Returns
        -------
        obs    : observation mới
        reward : phần thưởng
        done   : episode kết thúc chưa
        info   : thông tin thêm
        """
        assert action in [0, 1], "Action phải là 0 hoặc 1"
        assert self.current_item < self.n_items, "Episode đã kết thúc"

        reward = 0.0
        item = self.current_item
        valid_pick = False

        if action == 1:  # Chọn vật
            item_weights = self.weights[item]
            # Kiểm tra có vi phạm ràng buộc không
            if np.all(item_weights <= self.remaining_capacity):
                # Hợp lệ: cập nhật dung lượng và reward
                self.remaining_capacity -= item_weights
                self.total_value += self.values[item]
                self.selected.append(item)
                reward = float(self.values[item])
                valid_pick = True
            else:
                # Vi phạm ràng buộc → không chọn, không thưởng
                reward = 0.0

        # Chuyển sang vật tiếp theo
        self.current_item += 1
        done = (self.current_item >= self.n_items)
        obs = self._get_obs()

        info = {
            "item": item,
            "action": action,
            "valid_pick": valid_pick,
            "total_value": self.total_value,
            "selected_items": self.selected.copy(),
            "remaining_capacity": self.remaining_capacity.copy(),
        }

        return obs, reward, done, info

    def is_feasible(self, item_idx):
        """Kiểm tra vật item_idx có thể chọn được không"""
        return np.all(self.weights[item_idx] <= self.remaining_capacity)

    def get_feasible_actions(self):
        """Trả về danh sách action hợp lệ tại bước hiện tại"""
        if self.current_item >= self.n_items:
            return []
        if self.is_feasible(self.current_item):
            return [0, 1]  # có thể chọn hoặc bỏ
        else:
            return [0]     # chỉ có thể bỏ (vì chọn sẽ vi phạm)

    def render(self):
        """In thông tin trạng thái hiện tại"""
        print(f"\n{'='*45}")
        print(f"  Bước          : {self.current_item}/{self.n_items}")
        print(f"  Dung lượng CL : {self.remaining_capacity}")
        print(f"  Vật đã chọn   : {self.selected}")
        print(f"  Tổng giá trị  : {self.total_value:.2f}")
        print(f"{'='*45}")

    @property
    def observation_size(self):
        return self.obs_size

    @property
    def action_size(self):
        return 2  # 0: bỏ, 1: chọn

# Factory functions tạo môi trường

def make_random_mkp(n_items=10, n_constraints=3, seed=None):
    """
    Tạo bài toán MKP ngẫu nhiên

    Parameters
    ----------
    n_items       : số vật phẩm
    n_constraints : số ràng buộc (chiều)
    seed          : random seed để tái hiện kết quả
    """
    rng = np.random.default_rng(seed)

    values = rng.integers(1, 100, size=n_items).astype(np.float32)
    weights = rng.integers(1, 50, size=(n_items, n_constraints)).astype(np.float32)

    # Capacity = 50% tổng trọng lượng theo mỗi chiều
    capacity = (weights.sum(axis=0) * 0.5).astype(np.float32)

    return MKPEnv(n_items, n_constraints, capacity, values, weights)


def make_mkp_from_data(values, weights, capacity):
    """
    Tạo môi trường từ dữ liệu cho trước

    Parameters
    ----------
    values   : list giá trị, shape (n,)
    weights  : list trọng lượng, shape (n, k)
    capacity : list giới hạn, shape (k,)
    """
    values = np.array(values, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    capacity = np.array(capacity, dtype=np.float32)
    n_items = len(values)
    n_constraints = len(capacity)
    return MKPEnv(n_items, n_constraints, capacity, values, weights)


# Test nhanh môi trường


if __name__ == "__main__":
    print("=" * 50)
    print("   TEST MKP ENVIRONMENT")
    print("=" * 50)

    # Tạo bài toán mẫu nhỏ
    env = make_random_mkp(n_items=5, n_constraints=2, seed=42)

    print(f"\nBài toán:")
    print(f"  Số vật    : {env.n_items}")
    print(f"  Ràng buộc : {env.n_constraints} chiều")
    print(f"  Giá trị   : {env.values}")
    print(f"  Trọng lượng:\n{env.weights}")
    print(f"  Giới hạn  : {env.capacity}")

    # Chạy 1 episode với chiến lược random
    obs = env.reset()
    print(f"\nObs ban đầu (size={len(obs)}): {obs}")
    env.render()

    done = False
    step = 0
    while not done:
        feasible = env.get_feasible_actions()
        action = np.random.choice(feasible)  # random trong các action hợp lệ
        obs, reward, done, info = env.step(action)
        print(f"\nBước {step+1}: action={action} ({'chọn' if action==1 else 'bỏ'}) "
              f"| reward={reward:.1f} | valid={info['valid_pick']}")
        step += 1

    env.render()
    print(f"\nKết quả: chọn {len(env.selected)} vật, tổng giá trị = {env.total_value:.2f}")