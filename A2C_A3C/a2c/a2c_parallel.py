import time
import gymnasium as gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from .model import ActorCritic


def worker_process(worker_id, conn, env_name, gamma, t_max, obs_dim, act_dim):
    """
    병렬 워커:
    - 메인 프로세스에서 ('run', state_dict) 명령을 받으면
      최신 파라미터 로드 후 t_max step rollout 수행
    - (states, actions, rewards, next_states, dones) 를 메인으로 전송
    """
    env = gym.make(env_name)
    device = torch.device("cpu")

    # 로컬 모델 (CPU)
    local_model = ActorCritic(obs_dim, act_dim).to(device)
    local_model.eval()

    while True:
        cmd = conn.recv()
        if cmd[0] == "run":
            state_dict = cmd[1]
            local_model.load_state_dict(state_dict)

            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            obs, _ = env.reset()
            done = False
            step_count = 0

            while step_count < t_max and not done:
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = local_model(s)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(obs.astype(np.float32))
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_obs.astype(np.float32))
                dones.append(done)

                obs = next_obs
                step_count += 1

            # 메인 프로세스로 전송
            conn.send((states, actions, rewards, next_states, dones))

        elif cmd[0] == "close":
            env.close()
            conn.close()
            break
        else:
            # 정의되지 않은 명령
            pass


def train_parallel_a2c(
    env_name="CartPole-v1",
    n_workers=4,
    t_max=5,
    gamma=0.99,
    lr=1e-3,
    max_updates=500,
    use_gpu=False,
):
    # -----------------------------
    # 환경/모델 초기화 (메인 프로세스)
    # -----------------------------
    tmp_env = gym.make(env_name)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    tmp_env.close()

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device = {device}")

    global_model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(global_model.parameters(), lr=lr)

    # -----------------------------
    # 멀티프로세스 파이프 및 워커 생성
    # -----------------------------
    processes = []
    parent_conns = []
    for wid in range(n_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=worker_process,
            args=(wid, child_conn, env_name, gamma, t_max, obs_dim, act_dim),
        )
        p.daemon = True
        p.start()
        processes.append(p)
        parent_conns.append(parent_conn)

    # -----------------------------
    # 학습 완료 후 모델 저장
    # ----------------------------
    torch.save(global_model.state_dict(), "a2c_cartpole.pth")
    print("[INFO] Saved trained A2C model to a2c_cartpole.pth")
    
    # -----------------------------
    # 학습 루프 (Sync A2C)
    # -----------------------------
    for update in range(max_updates):

        # -------------------------
        #       SAMPLING
        # -------------------------
        t0 = time.time()

        # 모든 워커에게 "run" 명령 + 최신 파라미터 전달
        state_dict_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
        for conn in parent_conns:
            conn.send(("run", state_dict_cpu))

        # 워커들로부터 데이터 수집
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for conn in parent_conns:
            states, actions, rewards, next_states, dones = conn.recv()
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_rewards.extend(rewards)
            batch_next_states.extend(next_states)
            batch_dones.extend(dones)

        sampling_time = time.time() - t0

        # -------------------------
        #       TRAINING
        # -------------------------
        t1 = time.time()

        # numpy → torch 텐서
        states_t = torch.tensor(batch_states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(batch_actions, dtype=torch.int64, device=device)
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
        dones_t = torch.tensor(batch_dones, dtype=torch.float32, device=device)

        # 현재 상태 가치, 정책
        logits, values = global_model(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()

        # 다음 상태 가치 (bootstrap)
        with torch.no_grad():
            _, next_values = global_model(next_states_t)

        targets = rewards_t + gamma * next_values.squeeze(-1) * (1.0 - dones_t)
        advantages = targets - values.squeeze(-1)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropy.mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_time = time.time() - t1

        avg_reward = rewards_t.sum().item() / n_workers

        print(
            f"[UPDATE {update}] "
            f"avg_reward_per_worker={avg_reward:.2f}, "
            f"loss={loss.item():.3f}, "
            f"sampling={sampling_time:.3f}s, "
            f"training={training_time:.3f}s"
        )

    # -----------------------------
    # 워커 종료
    # -----------------------------
    for conn in parent_conns:
        conn.send(("close", None))
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # CPU
    # train_parallel_a2c(use_gpu=False)
    # GPU로 돌려보고 싶으면:
    train_parallel_a2c(use_gpu=True)
