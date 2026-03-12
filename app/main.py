import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def benchmark(sizes, repeats=10, seed=42):
    rng = np.random.default_rng(seed)

    all_runtimes = []
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

        runtimes = np.array(runtimes)
        mean_time = np.mean(runtimes)

        all_runtimes.append(runtimes)
        mean_times.append(mean_time)

        print(
            f"n={n:5d}, n^3={n**3:15d}, "
            f"mean_time={mean_time:.8f}s, "
            f"min={runtimes.min():.8f}s, max={runtimes.max():.8f}s"
        )

    return np.array(mean_times), all_runtimes


def plot_runtime_clusters_vs_n(sizes, all_runtimes, mean_times, out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_runtime_vs_n.png"
    filepath = os.path.join(out_dir, filename)

    plt.figure(figsize=(10, 6))

    # Plot all individual runtime samples as clustered scatter points
    for n, runtimes in zip(sizes, all_runtimes):
        x_vals = [n] * len(runtimes)
        plt.scatter(x_vals, runtimes, alpha=0.7, s=30)

    # Plot the mean runtime
    plt.plot(sizes, mean_times, marker="o", linewidth=2, label="Mean runtime")

    plt.xlabel("n")
    plt.ylabel("Runtime (seconds)")
    plt.title("SciPy linear_sum_assignment Runtime vs n\n(Individual Samples + Mean)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {filepath}")


if __name__ == "__main__":
    sizes = [2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 500, 1000, 2000, 3000]
    mean_times, all_runtimes = benchmark(sizes, repeats=50)
    plot_runtime_clusters_vs_n(sizes, all_runtimes, mean_times)