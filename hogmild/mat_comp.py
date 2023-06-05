import yaml
import pandas as pd
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


def main():
    config = load_default_config()
    config[cf.N_MOVIES] = 17770
    # config[cf.N_MOVIES] = 5128

    print("loading data...")
    data = load_data(config)
    num_samples = data.shape[0]
    print(f"loaded {num_samples} entries total")

    print("running simulation...")
    config[cf.NUM_SAMPLES] = num_samples
    cycle_counts, update_logs = get_update_logs(config)
    print("simulation finished")

    print(data)
    print(cycle_counts)
    print(update_logs)


if __name__ == "__main__":
    main()
