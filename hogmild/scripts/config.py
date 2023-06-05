import os

# Run the simulation, otherwise load the data
SIMULATION = "simulation"
# Number of samples to use for simulation only
NUM_SAMPLES = "num_samples"
# Whether to print the data associated with the data load or simulation
PRINT_DATA = "print_data"

# <<<< Common args across data sets and models >>>>
# Model hyper parameter initial learning rate
ALPHA_0 = "alpha_0"
# Model hyper parameter initial learning rate
DECAY_RATE = "decay_rate"
# Maximum number of epochs to run for
MAX_EPOCH = "max_epoch"
# When to stop training
STOPPING_CRITERION = "stopping_criterion"
# Whether or not to do hogwild
HOGWILD = "hogwild"
# The model to use
MODEL = "model"
# The dataset to use
DATASET = "dataset"
# RNG seed for weights initialization
RNG_SEED = "rng_seed"
# Number of banks to separate the weights into
N_WEIGHT_BANKS = "n_weight_banks"
# Number of worker threads in async sgd
N_WORKERS = "n_workers"
# Number of gradient folds that can happen in parallel
N_FOLDERS = "n_folders"
# Fifo depth in async sgd
FIFO_DEPTH = "fifo_depth"

# <<<< Timing Related >>>>
# Time to send a sample/update
SEND_DELAY = "send_delay"
# Time to deliver a sample/update
NETWORK_DELAY = "network_delay"
# Time to receive a sample/update
RECEIVE_DELAY = "receive_delay"
# Initiation interval of gradient calculation
GRADIENT_II = "gradient_ii"
# Latency of calculating one gradient
GRADIENT_LATENCY = "gradient_latency"
# Initiation interval of folding gradient updates
FOLD_II = "fold_ii"
# Latency of folding one gradient update
FOLD_LATENCY = "fold_latency"

# <<<< Matrix completion specific >>>>
# Number of features in the decomposition matrix
N_FEATURES = "n_features"
# Model hyper parameter mu
MU = "mu"
# Model hyper parameter lambda_xf
LAM_XF = "lam_xf"
# Model hyper parameter lambda_yf
LAM_YF = "lam_yf"
# Model hyper parameter lambda_xb
LAM_XB = "lam_xb"
# Model hyper parameter lambda_yb
LAM_YB = "lam_yb"

# <<<< Netflix dataset specific >>>>
# Number of movies to load
N_MOVIES = "n_movies"

HOGMILD_SIM_ARGS = [
    SIMULATION,
    NUM_SAMPLES,
    DATASET,
    PRINT_DATA,
    N_WEIGHT_BANKS,
    N_WORKERS,
    N_FOLDERS,
    FIFO_DEPTH,
    SEND_DELAY,
    NETWORK_DELAY,
    RECEIVE_DELAY,
    GRADIENT_II,
    GRADIENT_LATENCY,
    FOLD_II,
    FOLD_LATENCY,
    N_MOVIES,
]

CWD = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(CWD, "..", "configs", "default.yaml")
HOGMILD_SIM_PATH = os.path.join(
    CWD, "..", "hogmild_sim", "target", "release", "hogmild_sim"
)
