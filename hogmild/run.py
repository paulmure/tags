#!/usr/bin/python3

import os
import subprocess
import time
import yaml
import argparse
from collections import namedtuple
import matplotlib.pyplot as plt

import config as cf

Sample = namedtuple("Sample", "data_id weight_version")
Schedule = namedtuple("Schedule", "latency samples")

DAM_ROOT = os.environ["DAM_ROOT"]
SIMULATOR_PATH = os.path.join(
    DAM_ROOT, "bazel-bin", "apps", "hogmild", "hogmild_", "hogmild"
)

CWD = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CWD, "configs", "default.yaml")


def make_args_list(arg_names: list[str], prefix: str, config):
    res = []
    for arg in arg_names:
        res.append(f"{prefix}{arg}")
        res.append(str(config[arg]))
    return res


def run_simulator(config) -> Schedule:
    arg_names = [
        cf.FIFO_DEPTH,
        cf.FOLD_LATENCY,
        cf.GRADIENT_LATENCY,
        cf.NUM_SAMPLES,
        cf.NUM_WEIGHT_BANKS,
        cf.NUM_WORKERS,
        cf.NETWORK_DELAY,
        cf.SENDING_TIME,
    ]
    args = [SIMULATOR_PATH] + make_args_list(arg_names, cf.GO_PREFIX, config)
    output = subprocess.run(args, capture_output=True)
    return parse_schedule(output.stdout.decode("utf-8"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config_path", default=DEFAULT_CONFIG)
    return ap.parse_args()


def parse_schedule(schedule: str) -> Schedule:
    lines = schedule.splitlines()
    latency = int(lines[0])

    def parse_sample(line: str) -> Sample:
        toks = line.split(",")
        return Sample(data_id=int(toks[0]), weight_version=int(toks[1]))

    samples = list(map(lambda line: parse_sample(line), lines[1:]))

    return Schedule(latency=latency, samples=samples)


def plot_latency(xs: list[int], xlabel: str, latencies: list[int]):
    plt.clf()
    plt.loglog(xs, latencies)
    plt.xlabel(xlabel)
    plt.ylabel("Latency")
    plt.subplots_adjust(bottom=0.25, left=0.25)
    plt.show()


def sweep_num_workers(min: int, max: int, step: int, config):
    workers = []
    latencies = []
    for n in range(min, max, step):
        config[cf.NUM_WORKERS] = n
        config[cf.NUM_WEIGHT_BANKS] = n

        start_time = time.time()
        sched = run_simulator(config)
        end_time = time.time()
        print(
            f"nWorkers: {n},",
            f"CPU: {end_time - start_time:.2f} secs,",
            f"RDA: {sched.latency}",
        )

        workers.append(n)
        latencies.append(sched.latency)
    plot_latency(workers, "nWorkers", latencies)


def main():
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    sweep_num_workers(1, 16, 1, config)


if __name__ == "__main__":
    main()
