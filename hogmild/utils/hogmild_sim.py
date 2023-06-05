from subprocess import Popen, PIPE
from tqdm import tqdm

from . import config as cf


def make_hogmild_args_list(config):
    res = [cf.HOGMILD_SIM_PATH]
    for arg_name in cf.HOGMILD_SIM_ARGS:
        rust_flag = f"--{arg_name.replace('_', '-')}"
        arg_val = config[arg_name]
        if type(config[arg_name]) == bool:
            if arg_val:
                res.append(rust_flag)
        else:
            res.append(rust_flag)
            res.append(str(arg_val))
    return res


def run_hogmild(config) -> str:
    cmd = make_hogmild_args_list(config)
    p = Popen(cmd, stdout=PIPE, universal_newlines=True)

    output = ""

    still_loading = True
    while still_loading:
        byte = p.stdout.read(1)
        if byte.isnumeric():
            still_loading = False
            output += byte
            output += p.stdout.readline()
        else:
            print(output, end="")

    print("capturing output...")
    lines_remaining = int(output)
    with tqdm(total=lines_remaining) as pbar:
        for line in iter(p.stdout.readline, ""):
            output += line
            pbar.update(1)

    p.wait()
    return output.strip()
