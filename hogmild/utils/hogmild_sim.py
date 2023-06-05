import subprocess

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
    args = make_hogmild_args_list(config)
    output = subprocess.run(args, capture_output=True)
    output_str = output.stdout.decode("utf-8")
    return output_str.strip()
