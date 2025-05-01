import gymnasium
import torch


def scale(
    x: torch.Tensor,
    min_x: torch.Tensor,
    max_x: torch.Tensor,
    min_y: torch.Tensor,
    max_y: torch.Tensor,
) -> torch.Tensor:
    return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y


class LunarLanderV3:
    def __init__(self, gui: bool = False):
        self.env = gymnasium.make(
            "LunarLander-v3",
            render_mode="human" if gui else None,
            continuous=True,
            enable_wind=True,
        )
        self.env.reset(seed=42)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.action_min = torch.tensor(
            self.env.action_space.low,
            dtype=torch.float32,
        )
        self.action_max = torch.tensor(
            self.env.action_space.high,
            dtype=torch.float32,
        )

        self.state_min = torch.tensor(
            self.env.observation_space.low,
            dtype=torch.float32,
        )
        self.state_max = torch.tensor(
            self.env.observation_space.high,
            dtype=torch.float32,
        )

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = scale(
            obs,
            self.state_min,
            self.state_max,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        return obs, info

    def sample(self) -> torch.Tensor:
        action = self.env.action_space.sample()
        action = torch.tensor(action, dtype=torch.float32)
        action = scale(
            action,
            self.action_min,
            self.action_max,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        return action

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        float,
        bool,
        bool,
        dict,
    ]:
        action = scale(
            action,
            torch.tensor(-1.0),
            torch.tensor(1.0),
            self.action_min,
            self.action_max,
        ).numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = scale(
            obs,
            self.state_min,
            self.state_max,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
