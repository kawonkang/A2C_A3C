import torch.multiprocessing as mp
import torch.optim as optim
import gymnasium as gym
import torch
from .model import ActorCritic
from .worker import worker_process


def train_a3c_experiment(
    env_name="CartPole-v1",
    n_workers=4,
    gamma=0.99,
    t_max=5,
    lr=1e-4,
    max_episodes_per_worker=200,
    use_gpu=False,   # 추가: local 연산에 GPU 사용 여부
):
    """
    A3C 실험을 한 번 수행하는 함수.
    - 각 워커마다 max_episodes_per_worker 에피소드 수행 후 종료
    - use_gpu=True 인 경우 각 worker의 local 모델 forward/backward를 GPU에서 수행
    """

    tmp_env = gym.make(env_name)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    tmp_env.close()

    # Global model은 CPU에 두고 share_memory
    global_model = ActorCritic(obs_dim, act_dim)
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=lr)

    processes = []

    for wid in range(n_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                wid,
                global_model,
                optimizer,
                env_name,
                gamma,
                t_max,
                obs_dim,
                act_dim,
                max_episodes_per_worker,
                use_gpu,  # 여기 전달
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), "a3c_cartpole.pth")
    print("[INFO] Saved trained A3C model to a3c_cartpole.pth")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_a3c_experiment()
