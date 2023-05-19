#!/usr/bin/env python3

import os
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'training_set')
HEADER = 'User,Rating,Date\n'


def transform_data(file):
    file_path = os.path.join(DATA_DIR, file)

    with open(file_path, 'r') as fin:
        data = fin.read().splitlines(True)
    id = int(data[0][:-2])

    new_file_path = os.path.join(DATA_DIR, f'{id}.csv')
    with open(new_file_path, 'w') as fout:
        fout.writelines([HEADER] + data[1:])

    os.remove(file_path)


def main():
    files = os.listdir(DATA_DIR)
    with Pool() as pool:
        pool.map(transform_data, files)


if __name__ == '__main__':
    main()
