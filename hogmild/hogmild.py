from subprocess import Popen, PIPE

import config as cf


def make_hogmild_args_list(config):
    res = [cf.HOGMILD_PATH]
    for arg_name in cf.HOGMILD_ARGS:
        rust_flag = f"--{arg_name.replace('_', '-')}"
        arg_val = config[arg_name]
        if type(arg_val) == bool:
            if arg_val:
                res.append(rust_flag)
        else:
            res.append(rust_flag)
            res.append(str(arg_val))
    return res


def run_hogmild(config) -> tuple[int, list[float]]:
    cmd = make_hogmild_args_list(config)
    process = Popen(cmd, stdout=PIPE)

    stdout, _ = process.communicate()
    output = stdout.decode("utf-8").splitlines()

    cycles = int(output[0].split(":")[1])
    history = list(map(lambda x: float(x), output[1:]))

    return cycles, history
