import yaml
import pandas as pd
import numpy as np
from io import StringIO

from utils.hogmild_sim import run_hogmild
import utils.config as cf


def load_default_config():
    with open(cf.DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    config[cf.PRINT_DATA] = True
    return config


def get_hogmild_output(config) -> tuple[int, pd.DataFrame]:
    output_str = run_hogmild(config)

    print("parsing output into DataFrame...")
    first_line_idx = output_str.find("\n")
    count = int(output_str[:first_line_idx])

    header_idx = first_line_idx + 1
    data = StringIO(output_str[header_idx:])
    df = pd.read_csv(data, sep=",")

    return count, df


def load_data(config) -> pd.DataFrame:
    config[cf.SIMULATION] = False
    num_samples, df = get_hogmild_output(config)
    assert num_samples == df.shape[0]
    return df


def get_update_logs(config) -> tuple[int, pd.DataFrame]:
    config[cf.SIMULATION] = True
    cycle_count, df = get_hogmild_output(config)
    return cycle_count, df


class SparseMatrix:
    def __init__(self, config):
        print("loading data...")
        self.data = load_data(config)
        self.shape = (self.data["Row"].max() + 1, self.data["Column"].max() + 1)
        print(f"loaded {self.data.shape[0]} entries total, with shape {self.shape}")

    def __len__(self):
        return self.data.shape[0]


class DataLoader:
    def __init__(self, data: pd.DataFrame, update_logs: pd.DataFrame):
        self.update_logs = update_logs
        self.data = data
        self._versions = np.sort(np.array(pd.unique(self.update_logs.WeightVersion)))
        self._num_versions = self._versions.shape[0]

    def __iter__(self):
        self._curr_idx = 0
        return self

    def _next_samples(self) -> np.ndarray:
        curr_version = self._versions[self._curr_idx]
        curr_samples_idx = self.update_logs[
            self.update_logs.WeightVersion == curr_version
        ].index[0]

        self._curr_idx += 1
        if self._curr_idx < self._num_versions:
            next_version = self._versions[self._curr_idx]
            next_samples_idx = self.update_logs[
                self.update_logs.WeightVersion == next_version
            ].index[0]
        else:
            next_samples_idx = self.data.shape[0]

        samples = np.array(self.update_logs.SampleId[curr_samples_idx:next_samples_idx])
        return samples

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._curr_idx >= self._num_versions:
            raise StopIteration

        samples = self._next_samples()
        sub_df = self.data.iloc[samples]
        xs = np.array(sub_df["Row"])
        ys = np.array(sub_df["Column"])
        entries = np.array(sub_df["Entry"])
        return xs, ys, entries


def main():
    config = load_default_config()
    # config[cf.N_MOVIES] = 17770
    config[cf.N_MOVIES] = 8

    data = SparseMatrix(config)

    print("running simulation...")
    config[cf.NUM_SAMPLES] = len(data)
    cycle_counts, update_logs = get_update_logs(config)
    print("simulation finished")

    dataloader = DataLoader(data.data, update_logs)
    for x, y, e in dataloader:
        print(x, y, e)


if __name__ == "__main__":
    main()
