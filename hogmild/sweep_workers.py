import time
import yaml
import argparse
import matplotlib.pyplot as plt

from utils.hogmild_sim import run_hogmild
import utils.config as cf


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config_path", default=cf.DEFAULT_CONFIG)
    return ap.parse_args()


def plot_latency(xs: list[int], xlabel: str, latencies: list[int]):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    f.suptitle("Hardware Efficiency")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Cycles")
    ax1.plot(xs, latencies)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Cycles")
    ax2.loglog(xs, latencies)
    plt.show()


def sweep_num_workers(max: int, step: int, config):
    workers = []
    latencies = []
    config["simulation"] = True
    for i in range(0, max, step):
        n = i + 1
        config[cf.N_WORKERS] = n
        config[cf.N_WEIGHT_BANKS] = n
        config[cf.N_FOLDERS] = n

        start_time = time.time()
        output = run_hogmild(config)
        end_time = time.time()
        runtime = end_time - start_time

        latency = int(output)
        print(
            f"{n:>02d} workers,",
            f"CPU: {runtime:.2f} seconds,",
            f"RDU: {latency} cycles",
        )

        workers.append(n)
        latencies.append(latency)

    plot_latency(workers, "nWorkers", latencies)


def main():
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    sweep_num_workers(256, 1, config)


if __name__ == "__main__":
    main()
