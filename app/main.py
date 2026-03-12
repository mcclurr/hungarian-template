import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def cubic_proxy(rows, cols):
    return min(rows, cols) ** 2 * max(rows, cols)


def benchmark(shapes, repeats=10, seed=42):
    rng = np.random.default_rng(seed)

    labels = []
    proxies = []
    all_runtimes = []
    mean_times = []

    for rows, cols in shapes:
        runtimes = []

        # Pre-generate matrices so we're mostly timing the solver
        matrices = [rng.random((rows, cols)) for _ in range(repeats)]

        for cost in matrices:
            start = time.perf_counter()
            linear_sum_assignment(cost)
            end = time.perf_counter()
            runtimes.append(end - start)

        runtimes = np.array(runtimes)
        mean_time = np.mean(runtimes)
        proxy = cubic_proxy(rows, cols)
        label = f"{rows}x{cols}"

        labels.append(label)
        proxies.append(proxy)
        all_runtimes.append(runtimes)
        mean_times.append(mean_time)

        print(
            f"shape={label:>10}, "
            f"proxy={proxy:15d}, "
            f"mean_time={mean_time:.8f}s, "
            f"min={runtimes.min():.8f}s, "
            f"max={runtimes.max():.8f}s"
        )

    return labels, np.array(proxies), np.array(mean_times), all_runtimes


def plot_runtime_clusters_vs_proxy(labels, proxies, all_runtimes, mean_times, out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_runtime_vs_rectangular_proxy.png"
    filepath = os.path.join(out_dir, filename)

    plt.figure(figsize=(12, 7))

    # Plot all runtime samples as scatter clusters
    for label, proxy, runtimes in zip(labels, proxies, all_runtimes):
        x_vals = [proxy] * len(runtimes)
        plt.scatter(x_vals, runtimes, alpha=0.65, s=30)

    # Plot mean line
    order = np.argsort(proxies)
    sorted_proxies = proxies[order]
    sorted_means = mean_times[order]

    plt.plot(sorted_proxies, sorted_means, marker="o", linewidth=2, label="Mean runtime")

    # Annotate mean points with shape labels
    for i in order:
        plt.annotate(
            labels[i],
            (proxies[i], mean_times[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.xscale("log")

    plt.xlabel("n^3")
    plt.ylabel("Runtime (seconds)")
    plt.title("Hungarian Time Complexity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {filepath}")


if __name__ == "__main__":
    shapes = [
        (2, 2),
        (5, 5),
        (10, 10),
        (30, 30),
        (50, 50),
        (100, 100),
        (1000, 100),
        (1500, 1500),
        (2000, 2000),
        (3000, 3000),
        (4000, 4000),
        (5000, 5000)
    ]

    labels, proxies, mean_times, all_runtimes = benchmark(shapes, repeats=10)
    plot_runtime_clusters_vs_proxy(labels, proxies, all_runtimes, mean_times)