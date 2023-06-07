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


class MatComp:
    def __init__(self, data_loader: DataLoader, config, shape: tuple[int, int]):
        self.data_loader = data_loader

        self.alpha_0 = config[cf.ALPHA_0]
        self.decay_rate = config[cf.DECAY_RATE]
        self.max_epoch = config[cf.MAX_EPOCH]
        self.stopping_criterion = config[cf.STOPPING_CRITERION]

        self.n_features = config[cf.N_FEATURES]
        self.mu = config[cf.MU]
        self.lam_xf = config[cf.LAM_XF]
        self.lam_yf = config[cf.LAM_YF]
        self.lam_xb = config[cf.LAM_XB]
        self.lam_yb = config[cf.LAM_YB]

        self.shape = shape

        self.X = np.random.randn(self.shape[0], self.n_features)
        self.Y = np.random.randn(self.shape[1], self.n_features)
        self.XB = np.random.randn(self.shape[0])
        self.YB = np.random.randn(self.shape[1])

    def total_loss(self) -> float:
        loss = 0
        for xs, ys, elems in self.data_loader:
            x_batch = self.X[xs]
            y_batch = self.Y[ys]
            xb_batch = self.XB[xs]
            yb_batch = self.YB[ys]

            pred = np.sum(x_batch * y_batch, axis=1) + xb_batch + yb_batch + self.mu
            error_square = np.square(elems - pred)

            x_regu = self.lam_xf * np.sum(np.square(x_batch), axis=1)
            y_regu = self.lam_yf * np.sum(np.square(y_batch), axis=1)
            xb_regu = self.lam_xb * xb_batch
            yb_regu = self.lam_yb * yb_batch

            loss += np.sum(error_square + x_regu + y_regu + xb_regu + yb_regu)
        return loss

    def train(self) -> list[float]:
        history = [self.total_loss()]
        print(history[-1])
        for i in range(self.max_epoch):
            curr_loss = 0
            learning_rate = (1 / (1 + self.decay_rate * i)) * self.alpha_0
            for xs, ys, elems in self.data_loader:
                e_shape = (xs.shape[0], 1)
                x_batch = self.X[xs]
                y_batch = self.Y[ys]
                xb_batch = self.XB[xs]
                yb_batch = self.YB[ys]

                pred = np.sum(x_batch * y_batch, axis=1) + xb_batch + yb_batch + self.mu
                pred = np.reshape(pred, e_shape)
                elems = np.reshape(elems, e_shape)
                error = elems - pred

                x_regu = np.reshape(
                    self.lam_xf * np.sum(np.square(x_batch), axis=1), e_shape
                )
                y_regu = np.reshape(
                    self.lam_yf * np.sum(np.square(y_batch), axis=1), e_shape
                )
                xb_regu = self.lam_xb * xb_batch
                yb_regu = self.lam_yb * yb_batch

                curr_loss += np.sum(
                    np.square(error) + x_regu + y_regu + xb_regu + yb_regu
                )

                self.X[xs] += learning_rate * (error * y_batch - x_regu * x_batch)
                self.Y[ys] += learning_rate * (error * x_batch - y_regu * y_batch)
                self.XB[xs] += np.reshape(learning_rate * (error - xb_regu), e_shape)
                self.YB[ys] += np.reshape(learning_rate * (error - yb_regu), e_shape)

            last_loss = history[-1]
            history.append(curr_loss)
            print(curr_loss)
            if (last_loss - curr_loss) / last_loss < self.stopping_criterion:
                break

        return history


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
    model = MatComp(dataloader, config, data.shape)
    history = model.train()


if __name__ == "__main__":
    main()
