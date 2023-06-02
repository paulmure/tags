#!/usr/bin/python3

import subprocess
import time
import yaml
import argparse
import matplotlib.pyplot as plt

import config as cf


def make_hogmild_args_list(config):
    res = [cf.HOGMILD_PATH]
    for arg in cf.RUST_ARGS:
        rust_flag = f"--{arg.replace('_', '-')}"
        if type(config[arg]) == bool:
            if arg:
                res.append(rust_flag)
        else:
            res.append(rust_flag)
            res.append(str(config[arg]))
    return res


def run_simulator(config) -> str:
    args = make_hogmild_args_list(config)
    output = subprocess.run(args, capture_output=True)
    return output.stdout.decode("utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config_path", default=cf.DEFAULT_CONFIG)
    return ap.parse_args()


def plot_latency(xs: list[int], xlabel: str, latencies: list[int]):
    plt.figure(figsize=(14, 7))
    plt.loglog(xs, latencies)
    plt.xlabel(xlabel)
    plt.ylabel("Latency")
    plt.show()


def sweep_num_workers(min: int, max: int, step: int, config):
    workers = []
    latencies = []
    config["simulation_only"] = True
    for n in range(min, max, step):
        config[cf.N_WORKERS] = n
        config[cf.N_WEIGHT_BANKS] = n
        config[cf.N_FOLDERS] = n

        start_time = time.time()
        output = run_simulator(config)
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
    sweep_num_workers(1, 129, 1, config)


if __name__ == "__main__":
    main()
