import torch
from model import PolicyNet
from env import LunarLanderV3


def eval(loop: int = 0):
    env = LunarLanderV3(gui=True)
    state, info = env.reset()

    policy_net = PolicyNet(
        action_dim=env.action_dim,
        state_dim=env.state_dim,
    )
    policy_net.eval()

    checkpoint = torch.load(f"checkpoints/checkpoint{loop}.pth", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.to(device)

    for episode in range(100):
        state, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                _, mean, _ = policy_net.get_action(state.to(device))
                action = mean.detach().cpu()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state


if __name__ == "__main__":
    eval(16)
