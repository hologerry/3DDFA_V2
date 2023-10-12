import os

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def print_cmd(cmd):
    print(cmd)
    os.system(cmd)


def one_job_process(args):
    available_gpu = torch.cuda.device_count()
    cmds = []
    for gpu_idx in range(args.gpu_num):
        cmd = f"export CUDA_VISIBLE_DEVICES={gpu_idx%available_gpu} && python vfhq_forward_all.py --job_idx {args.job_idx} --job_num {args.job_num} --gpu_idx {gpu_idx} --gpu_num {args.gpu_num}"
        if args.debug:
            cmd += " --debug"
        cmds.append(cmd)

    pool = Pool(args.gpu_num)
    pool.map(print_cmd, cmds)
    pool.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=4)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    print(f"job {args.job_idx} started")
    one_job_process(args)
