#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
from pathlib import Path

file = sys.argv[1]

with open(file, "r") as f:
    lines = f.readlines()
history = list(map(lambda x: float(x), lines))
xs = list(range(len(history)))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=400)
f.suptitle("Netflix Matrix Completion")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")

ax1.plot(xs, history)
ax2.loglog(xs, history)

plt.savefig(f"{Path(file).stem}.png")
