"""Drug Dose Environment - 1-compartment PK control using Gymnasium API."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DrugDoseEnv(gym.Env):
    """
    Toy 1-compartment PK control environment (Gymnasium API).

    Observation: [C] - drug concentration
    Action: [dose] in [0, max_dose]
    Dynamics: C <- C*(1-k) + dose + noise
    Reward: +1 in [c_min, c_max], else negative distance to therapeutic window
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 100,
        k: float = 0.05,
        max_dose: float = 1.0,
        c_min: float = 2.0,
        c_max: float = 3.0,
        noise_std: float = 0.0,
        randomize_k: bool = False,
        k_range: tuple[float, float] = (0.03, 0.08),
        render_mode: str | None = None,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.k_base = float(k)
        self.max_dose = float(max_dose)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.noise_std = float(noise_std)
        self.randomize_k = bool(randomize_k)
        self.k_range = (float(k_range[0]), float(k_range[1]))
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_dose], dtype=np.float32),
            dtype=np.float32,
        )

        self.t = 0
        self.C = 0.0
        self.k = self.k_base
        self._np_random = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.t = 0
        self.C = 0.0
        if self.randomize_k:
            self.k = float(self.np_random.uniform(self.k_range[0], self.k_range[1]))
        else:
            self.k = self.k_base

        obs = np.array([self.C], dtype=np.float32)
        info = {"C": self.C, "k": self.k}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.t += 1
        dose = float(np.clip(action[0], 0.0, self.max_dose))

        noise = 0.0
        if self.noise_std > 0:
            noise = float(self.np_random.normal(0.0, self.noise_std))

        self.C = max(0.0, self.C * (1.0 - self.k) + dose + noise)

        # Reward: +1 in therapeutic window, else negative distance
        if self.c_min <= self.C <= self.c_max:
            reward = 1.0
        elif self.C < self.c_min:
            reward = -(self.c_min - self.C)
        else:
            reward = -(self.C - self.c_max)

        terminated = False  # Episode doesn't terminate early
        truncated = self.t >= self.max_steps  # Truncated when max steps reached

        obs = np.array([self.C], dtype=np.float32)
        info = {"C": self.C, "dose": dose, "k": self.k}

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the environment (no-op for this simple env)."""
        if self.render_mode == "human":
            print(f"Step {self.t}: C={self.C:.3f}, k={self.k:.4f}")
