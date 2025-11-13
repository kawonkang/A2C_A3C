# demo_cartpole.py
import time
import gymnasium as gym
import torch
import torch.nn.functional as F

from a2c.model import ActorCritic  # 경로는 프로젝트 구조에 맞게 조정


def load_model(model_path, env_name="CartPole-v1", device="cpu"):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    model = ActorCritic(obs_dim, act_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def demo(model_path="a2c_cartpole.pth", env_name="CartPole-v1", n_episodes=5):
    device = torch.device("cpu")

    # 렌더링용 환경 (human 모드)
    env = gym.make(env_name, render_mode="human")
    model = load_model(model_path, env_name, device)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()  # greedy policy

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 사람이 눈으로 보게 약간 천천히
            time.sleep(0.02)

        print(f"[EP {ep}] total_reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    demo()
