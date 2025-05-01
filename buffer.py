from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class EpisodeBufferParams:
    gamma: float
    gae_lambda: float


class EpisodeBuffer:
    def __init__(self, params: EpisodeBufferParams) -> None:
        self.params = params

        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.next_states: list[torch.Tensor] = []
        self.done_masks: list[bool] = []
        self.action_log_probs: list[float] = []

        self.advantages: list[float] = []
        self.value_targets: list[float] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        action_log_prob: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done_masks.append(done)
        self.action_log_probs.append(action_log_prob)

    def compute_advantages_and_targets(
        self,
        value_network: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(value_network.parameters()).device

        with torch.no_grad():
            values = (
                value_network(torch.stack(self.states).to(device))
                .detach()
                .cpu()
                .squeeze(1)
            )
            next_values = (
                value_network(torch.stack(self.next_states).to(device))
                .detach()
                .cpu()
                .squeeze(1)
            )

        self.advantages.clear()
        self.value_targets.clear()

        last_adv = 0
        for i in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[i]
                + (self.params.gamma * next_values[i] * (1 - int(self.done_masks[i])))
            ) - values[i]
            last_adv = delta + self.params.gamma * self.params.gae_lambda * (
                last_adv * (1 - int(self.done_masks[i]))
            )
            value_target = last_adv + values[i]
            self.advantages.insert(0, float(last_adv))
            self.value_targets.insert(0, float(value_target))


@dataclass
class EpisodeDatasetParams:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool


class EpisodeDataset(Dataset):
    def __init__(self, params: EpisodeDatasetParams) -> None:
        super().__init__()

        self.params = params

        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.action_log_probs: list[torch.Tensor] = []
        self.advantages: list[torch.Tensor] = []
        self.value_targets: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.states[index],
            self.actions[index],
            self.action_log_probs[index],
            self.advantages[index],
            self.value_targets[index],
        )

    def add(
        self,
        episode_buffer: EpisodeBuffer,
    ) -> None:
        self.states.extend(episode_buffer.states)
        self.actions.extend(episode_buffer.actions)
        self.action_log_probs.extend(
            [
                torch.tensor(log_prob, dtype=torch.float32)
                for log_prob in episode_buffer.action_log_probs
            ]
        )
        self.advantages.extend(
            [
                torch.tensor(advantage, dtype=torch.float32)
                for advantage in episode_buffer.advantages
            ]
        )
        self.value_targets.extend(
            [
                torch.tensor([value_target], dtype=torch.float32)
                for value_target in episode_buffer.value_targets
            ]
        )

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.advantages.clear()
        self.value_targets.clear()

    def create_dataloader(
        self,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
        )
