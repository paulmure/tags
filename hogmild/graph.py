import sys
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple

History = namedtuple("History", "name xs history")


def load_history(file: str) -> History:
    with open(file, "r") as f:
        lines = f.readlines()

    name = Path(file).stem
    history = list(map(lambda x: float(x.split(",")[0]), lines))
    xs = list(range(len(history)))

    return History(name, xs, history)


def plot_histories(hists: list[History]):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=400)
    f.suptitle("Netflix Matrix Completion")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    list(map(lambda h: ax1.plot(h.xs, h.history, label=h.name), hists))
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    list(map(lambda h: ax2.loglog(h.xs, h.history, label=h.name), hists))
    ax2.legend(loc="upper right")

    names = list(map(lambda h: h.name, hists))
    output_file = f"{'_'.join(names)}.png"
    plt.savefig(output_file)


def main():
    files = sys.argv[1:]
    histories = list(map(load_history, files))
    plot_histories(histories)


if __name__ == "__main__":
    main()
