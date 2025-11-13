import time
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F

from .model import ActorCritic


def compute_returns(rewards, values, gamma, next_value, done):
    R = next_value * (1 - done)
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    returns = torch.stack(returns)
    advantages = returns - torch.stack(values)
    return returns, advantages


def train_single(env_name="CartPole-v1", lr=1e-3, gamma=0.99, max_episodes=500):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(max_episodes):

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        log_probs, values, rewards = [], [], []
        done = False

        # -------------------------
        #       SAMPLING
        # -------------------------
        t0 = time.time()

        while not done:
            action, log_prob, value = model.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

            obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

        sampling_time = time.time() - t0

        # -------------------------
        #       TRAINING
        # -------------------------
        t1 = time.time()

        # bootstrap value
        next_value = torch.tensor(0.0, device=device)

        returns, advantages = compute_returns(rewards, values, gamma, next_value, done)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_time = time.time() - t1

        print(f"[EP {ep}] reward={sum(r.item() for r in rewards)}, "
              f"sampling={sampling_time:.3f}s, training={training_time:.3f}s")


if __name__ == "__main__":
    train_single()
