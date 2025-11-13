import time
import csv
import os

from a2c.a2c_parallel import train_parallel_a2c
from a3c.a3c_train import train_a3c_experiment

RESULTS_FILE = "results.csv"


def run_a2c_experiment(name, use_gpu, max_updates=200, n_workers=4):
    print(f"\n===== Running {name} (A2C, use_gpu={use_gpu}) =====")
    t0 = time.time()
    train_parallel_a2c(
        env_name="CartPole-v1",
        n_workers=n_workers,
        t_max=5,
        gamma=0.99,
        lr=1e-3,
        max_updates=max_updates,
        use_gpu=use_gpu,
    )
    total_time = time.time() - t0
    print(f"===== {name} finished in {total_time:.2f} seconds =====")
    return total_time


def run_a3c_experiment(name, use_gpu, n_workers=4, max_episodes_per_worker=200):
    print(f"\n===== Running {name} (A3C, use_gpu={use_gpu}) =====")
    t0 = time.time()
    train_a3c_experiment(
        env_name="CartPole-v1",
        n_workers=n_workers,
        gamma=0.99,
        t_max=5,
        lr=1e-4,
        max_episodes_per_worker=max_episodes_per_worker,
        use_gpu=use_gpu,
    )
    total_time = time.time() - t0
    print(f"===== {name} finished in {total_time:.2f} seconds =====")
    return total_time


def main():
    experiments = [
        # A2C Sync
        {"name": "A2C_CPU", "algo": "A2C", "mode": "sync", "use_gpu": False},
        {"name": "A2C_GPU", "algo": "A2C", "mode": "sync", "use_gpu": True},
        # A3C Async
        {"name": "A3C_CPU", "algo": "A3C", "mode": "async", "use_gpu": False},
        {"name": "A3C_GPU", "algo": "A3C", "mode": "async", "use_gpu": True},
    ]

    rows = []

    for exp in experiments:
        if exp["algo"] == "A2C":
            total_time = run_a2c_experiment(
                name=exp["name"],
                use_gpu=exp["use_gpu"],
                max_updates=200,
                n_workers=4,
            )
            scale = 200  # update 수 기준
        elif exp["algo"] == "A3C":
            total_time = run_a3c_experiment(
                name=exp["name"],
                use_gpu=exp["use_gpu"],
                n_workers=4,
                max_episodes_per_worker=200,
            )
            scale = 4 * 200  # 워커 수 × 에피소드 수 (대략적 규모)
        else:
            continue

        rows.append(
            {
                "name": exp["name"],
                "algo": exp["algo"],
                "mode": exp["mode"],
                "use_gpu": exp["use_gpu"],
                "n_workers": 4,
                "scale": scale,
                "total_time_sec": total_time,
            }
        )

    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "algo",
                "mode",
                "use_gpu",
                "n_workers",
                "scale",
                "total_time_sec",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[INFO] Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
