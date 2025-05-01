import os
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

import wandb

from env import LunarLanderV3
from model import PolicyNet, ValueNet
from buffer import (
    EpisodeDataset,
    EpisodeDatasetParams,
    EpisodeBuffer,
    EpisodeBufferParams,
)


@dataclass
class TrainParams:
    global_loops: int
    episodes_per_loop: int
    max_steps_per_episode: int
    epochs_per_loop: int
    clip_epsilon: float
    policy_lr: float = 0.001
    value_lr: float = 0.001


def train():
    train_params = TrainParams(
        global_loops=100000,
        episodes_per_loop=100,
        max_steps_per_episode=10000,
        epochs_per_loop=20,
        clip_epsilon=0.2,
    )
    dataset_params = EpisodeDatasetParams(
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    episode_buffer_params = EpisodeBufferParams(
        gamma=0.99,
        gae_lambda=0.9,
    )

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if WANDB_API_KEY is None:
        raise ValueError(
            "WANDB_API_KEY environment variable is not set. "
            "Please set it to your Weights & Biases API key."
        )
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="lunar-lander-ppo-pt",
        config={
            "train_params": asdict(train_params),
            "dataset_params": asdict(dataset_params),
            "episode_buffer_params": asdict(episode_buffer_params),
        },
    )

    env = LunarLanderV3(gui=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = PolicyNet(env.action_dim, env.state_dim).to(device)
    value_net = ValueNet(env.state_dim).to(device)

    policy_net_optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=3e-4,
    )

    value_net_optimizer = torch.optim.Adam(
        value_net.parameters(),
        lr=1e-3,
    )

    dataset = EpisodeDataset(
        params=EpisodeDatasetParams(
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    )

    for loop in range(train_params.global_loops):
        dataset.clear()
        # Collect episodes
        rewards_history = []
        for episode in range(train_params.episodes_per_loop):
            episode_buffer = EpisodeBuffer(
                params=EpisodeBufferParams(
                    gamma=0.99,
                    gae_lambda=0.9,
                ),
            )
            state, info = env.reset()
            total_reward = 0
            for step in range(train_params.max_steps_per_episode):
                with torch.no_grad():
                    action, mean, log_prob = policy_net.get_action(state.to(device))
                    action = action.detach().cpu()
                    log_prob = log_prob.detach().cpu()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                episode_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    action_log_prob=log_prob,
                )
                state = next_state
                if done:
                    break
            episode_buffer.compute_advantages_and_targets(value_net)
            dataset.add(episode_buffer)
            print(
                f"Loop: {loop}, Episode: {episode}, Steps: {step}, Reward: {total_reward}"
            )
            rewards_history.append(total_reward)

        # Create dataloader
        dataloader = dataset.create_dataloader()

        # Train
        policy_loss_avg_history = []
        value_loss_avg_history = []

        for epoch in range(train_params.epochs_per_loop):
            policy_loss_history = []
            value_loss_history = []
            for (
                states,
                actions,
                action_log_probs,
                advantages,
                value_targets,
            ) in dataloader:
                states = states.to(device)
                actions = actions.to(device)
                action_log_probs = action_log_probs.to(device)
                advantages = advantages.to(device)
                value_targets = value_targets.to(device)

                # Update policy network
                dist, mean = policy_net.get_distribution(states)
                new_log_prob = dist.log_prob(actions)
                ratio = torch.exp(new_log_prob - action_log_probs)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio,
                    1 - train_params.clip_epsilon,
                    1 + train_params.clip_epsilon,
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                policy_net_optimizer.zero_grad()
                policy_loss.backward()
                policy_net_optimizer.step()
                policy_loss_history.append(policy_loss.item())

                # Update value network
                values = value_net(states)
                value_loss = F.mse_loss(values, value_targets)
                value_net_optimizer.zero_grad()
                value_loss.backward()
                value_net_optimizer.step()
                value_loss_history.append(value_loss.item())

            policy_loss_avg_history.append(
                sum(policy_loss_history) / len(policy_loss_history)
            )
            value_loss_avg_history.append(
                sum(value_loss_history) / len(value_loss_history)
            )
            print(
                f"Loop: {loop}, Epoch: {epoch}, Policy Loss: {policy_loss_avg_history[-1]}, Value Loss: {value_loss_avg_history[-1]}"
            )

        # Log to wandb
        wandb.log(
            {
                "policy_loss": sum(policy_loss_avg_history)
                / len(policy_loss_avg_history),
                "value_loss": sum(value_loss_avg_history) / len(value_loss_avg_history),
                "average_reward": sum(rewards_history) / len(rewards_history),
                "min_reward": min(rewards_history),
                "max_reward": max(rewards_history),
            }
        )

        if not os.path.exists(f"checkpoints"):
            os.makedirs(f"checkpoints", exist_ok=True)

        torch.save(
            {
                "loop": loop,
                "rewards_history": rewards_history,
                "policy_loss_history": policy_loss_avg_history,
                "value_loss_history": value_loss_avg_history,
                "policy_net": policy_net.state_dict(),
                "value_net": value_net.state_dict(),
                "policy_net_optimizer": policy_net_optimizer.state_dict(),
                "value_net_optimizer": value_net_optimizer.state_dict(),
            },
            f"checkpoints/checkpoint{loop}.pth",
        )


if __name__ == "__main__":
    train()
