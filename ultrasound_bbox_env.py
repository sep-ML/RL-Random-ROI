import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class ManualBBoxEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, image_size=256, step_frac=0.1, max_steps=50):
        super().__init__()

        self.image_size = image_size
        self.step_frac = step_frac
        self.max_steps = max_steps

        self.step_penalty = -0.01

        # ---- Action space (8 discrete actions) ----
        self.action_space = spaces.Discrete(9)

        # ---- Observation space (normalized box) ----
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    # --------------------------------------------------
    # Core Gym API
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        
        self.done = False

        self.image = np.random.rand(self.image_size, self.image_size)

        self.gt_box = self._random_box()
        self.box = self._random_box()

        #self.prev_iou = self.compute_iou(self.box, self.gt_box)

        return self._get_obs(), {}

    def step(self, action):

        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        self.steps += 1

        x, y, w, h = self.box
        dx = self.step_frac * w
        dy = self.step_frac * h

        # ---- Movement actions ----
        if action == 0:      # up
            y -= dy
        elif action == 1:    # down
            y += dy
        elif action == 2:    # left
            x -= dx
        elif action == 3:    # right
            x += dx
        elif action == 4:    # increase width
            w *= 1.1
        elif action == 5:    # decrease width
            w *= 0.9
        elif action == 6:    # increase height
            h *= 1.1
        elif action == 7:    # decrease height
            h *= 0.9
                # Stop action
        elif action == 8:
            terminated = True
            truncated = False
            reward = iou
            if iou > .9 :
                reward += 0.5
            self.done = True
            return self._get_obs(), reward, terminated, truncated, {}

        # ---- Clamp boundaries ----
        w = np.clip(w, 20, self.image_size)
        h = np.clip(h, 20, self.image_size)

        x = np.clip(x, w / 2, self.image_size - w / 2)
        y = np.clip(y, h / 2, self.image_size - h / 2)

        self.box = np.array([x, y, w, h])

        # ---- Reward ----
        iou = self.compute_iou(self.box, self.gt_box)


        reward = iou + self.step_penalty
        self.prev_iou = iou
        
        if iou > .9 :
            reward += 0.5

        terminated = False
        truncated = False

      

        # Max steps
        if self.steps >= self.max_steps:
            
            truncated = True
            

        if terminated or truncated:
            self.done = True

        return self._get_obs(), reward, terminated, truncated, {}

    # --------------------------------------------------
    # Observation (Normalized State)
    # --------------------------------------------------

    def _get_obs(self):
        x, y, w, h = self.box

        return np.array([
            x / self.image_size,
            y / self.image_size,
            w / self.image_size,
            h / self.image_size
        ], dtype=np.float32)

    # --------------------------------------------------
    # Utility Functions
    # --------------------------------------------------

    def compute_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)

        intersection = inter_w * inter_h
        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _random_box(self):
        w = np.random.randint(40, 80)
        h = np.random.randint(40, 80)

        x = np.random.randint(w // 2, self.image_size - w // 2)
        y = np.random.randint(h // 2, self.image_size - h // 2)

        return np.array([x, y, w, h], dtype=float)

    # --------------------------------------------------
    # Rendering
    # --------------------------------------------------

    def render(self, ax):
        ax.clear()
        ax.imshow(self.image, cmap="gray")
        self._draw_box(ax, self.box, color="blue")
        self._draw_box(ax, self.gt_box, color="red")
        
        ax.set_title("Blue: Agent | Red: Ground Truth")

    def _draw_box(self, ax, box, color="blue"):
        x, y, w, h = box
        x1 = x - w / 2
        y1 = y - h / 2

        rect = plt.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)