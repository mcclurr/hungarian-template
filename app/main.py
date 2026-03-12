import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def benchmark(sizes, repeats=10, seed=42):
    rng = np.random.default_rng(seed)

    n_cubed = []
    mean_times = []

    for n in sizes:
        runtimes = []

        # Pre-generate matrices so we're mostly timing the solver
        matrices = [rng.random((n, n)) for _ in range(repeats)]

        for cost in matrices:
            start = time.perf_counter()
            linear_sum_assignment(cost)
            end = time.perf_counter()
            runtimes.append(end - start)

        mean_time = np.mean(runtimes)

        n_cubed.append(n ** 3)
        mean_times.append(mean_time)

        print(f"n={n:3d}, n^3={n**3:9d}, mean_time={mean_time:.8f}s")

    return np.array(n_cubed), np.array(mean_times)


def plot_runtime_vs_n3(n_cubed, mean_times, out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_runtime_vs_n3.png"
    filepath = os.path.join(out_dir, filename)

    plt.figure(figsize=(8, 5))
    plt.plot(n_cubed, mean_times, marker="o")
    plt.xlabel("n^3")
    plt.ylabel("Runtime (seconds)")
    plt.title("SciPy linear_sum_assignment Runtime vs n^3")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {filepath}")


if __name__ == "__main__":
    sizes = [2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    n_cubed, mean_times = benchmark(sizes, repeats=10)
    plot_runtime_vs_n3(n_cubed, mean_times)