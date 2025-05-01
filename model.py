import torch
from torch import nn
from torch.functional import F
from torchrl.modules import TanhNormal


class PolicyNet(nn.Module):
    def __init__(self, action_dim: int, state_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)

        self.linear_mean = nn.Linear(128, action_dim)
        self.linear_log_std = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        mean = self.linear_mean(x)
        log_std = self.linear_log_std(x)
        log_std = torch.clamp(log_std, -20, 0.7)
        return mean, log_std

    def get_distribution(
        self,
        state: torch.Tensor,
    ) -> tuple[
        TanhNormal,
        torch.Tensor,
    ]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = TanhNormal(mean, std)
        mean_tanh = torch.tanh(mean)
        return dist, mean_tanh

    def get_action(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        dist, mean = self.get_distribution(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, mean, log_prob


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear_value = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        value = self.linear_value(x)
        return value
