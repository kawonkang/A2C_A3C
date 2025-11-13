import time
import gymnasium as gym
import torch
import torch.nn.functional as F


def worker_process(
    worker_id,
    global_model,
    optimizer,
    env_name,
    gamma,
    t_max,
    obs_dim,
    act_dim,
    max_episodes,  # 워커당 최대 에피소드 수
    use_gpu,       # 추가: GPU 사용 여부
):
    """
    A3C worker:
    - 독립 CartPole 환경
    - rollout + local backward
    - gradient를 global model에 반영
    - max_episodes 만큼 에피소드 수행 후 종료
    - use_gpu=True 일 경우 local forward/backward를 GPU에서 수행
      (global model과 optimizer는 CPU에 유지)
    """

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[Worker {worker_id}] device = {device}")

    env = gym.make(env_name)

    # local 모델은 worker 쪽에서만 사용
    local_model = type(global_model)(obs_dim, act_dim).to(device)
    # global(CPU) → local(device)로 파라미터 복사
    local_model.load_state_dict(global_model.state_dict())

    episode = 0

    while episode < max_episodes:

        states, actions, rewards, values = [], [], [], []

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        t0 = time.time()

        for _ in range(t_max):
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = local_model(s)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(s)
            actions.append(torch.tensor(action, dtype=torch.int64, device=device))
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            values.append(value.squeeze())

            ep_reward += reward
            obs = next_obs
            if done:
                break

        sampling_time = time.time() - t0

        # ----- bootstrap -----
        if done:
            R = torch.tensor(0.0, device=device)
        else:
            ns = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, nxt_value = local_model(ns)
            R = nxt_value.squeeze()

        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns.reverse()
        returns = torch.stack(returns)
        values = torch.stack(values)
        advantages = returns - values

        # ----- local backward -----
        logits, _ = local_model(torch.cat(states))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(torch.stack(actions))
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        t1 = time.time()
        loss.backward()
        training_time = time.time() - t1

        # ----- local grad → global(CPU) grad 복사 -----
        # global_model 은 CPU(shared_memory)에 있다고 가정
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad.detach().cpu()

        optimizer.step()

        # global(CPU) → local(device) sync
        local_model.load_state_dict(global_model.state_dict())

        print(
            f"[Worker {worker_id}] "
            f"ep={episode}, "
            f"reward={ep_reward:.1f}, "
            f"loss={loss.item():.3f}, "
            f"sampling={sampling_time:.3f}s, "
            f"training={training_time:.3f}s"
        )

        episode += 1

    env.close()
