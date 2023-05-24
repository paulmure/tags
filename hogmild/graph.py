#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt

file = sys.argv[1]

with open(file, "r") as f:
    lines = f.readlines()
history = list(map(lambda x: float(x), lines))

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.plot(range(len(history)), history)
plt.savefig("loss.png")
