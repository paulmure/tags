import copy
import yaml
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from collections import namedtuple

from hogmild import run_hogmild
import config as cf


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


TestResult = namedtuple("TestResult", "n_workers cycles history")


def test_worker(config) -> TestResult:
    n_workers = config[cf.N_WORKERS]
    cycles, history = run_hogmild(config)
    return TestResult(n_workers, cycles, history)


def sweep_num_workers(max: int, step: int, config):
    config["simulation"] = False
    configs = []
    for i in range(0, max, step):
        n = i + 1
        config[cf.N_WORKERS] = n
        config[cf.N_WEIGHT_BANKS] = n
        config[cf.N_FOLDERS] = n
        configs.append(copy.deepcopy(config))

    with Pool() as pool:
        results = list(tqdm(pool.imap(test_worker, configs), total=len(configs)))

    with open("sweep_workers.p", "wb") as f:
        pickle.dump(results, f)


def main():
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    sweep_num_workers(32, 1, config)


if __name__ == "__main__":
    main()
