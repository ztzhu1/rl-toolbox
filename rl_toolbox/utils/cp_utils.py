from copy import deepcopy
import os
import pathlib
import re

import json
import keyboard
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import trange

import gymnasium as gym
import torch
from torch import nn
from torch.optim import Optimizer

import safetensors  # TODO

from .backend import get_device
from .network_utils import as_tensor32


def search_cp_file_name(cp_dir, epochs=None):
    files = os.listdir(cp_dir)
    if len(files) == 0:  # empty
        return

    r = re.compile(r"epochs_(\d+).pt")
    all_epochs = []
    for filename in files:
        result = r.fullmatch(filename)
        if result is None:
            continue
        e = int(result.groups()[0])
        if e == epochs:
            return os.path.join(cp_dir, result.string)
        if epochs is None:
            all_epochs.append(e)
    if epochs is not None:
        return  # not found

    epochs = np.max(all_epochs)
    return os.path.join(cp_dir, f"epochs_{epochs}.pt")


def save_checkpoint(cp_dir, epochs, config, args_in_name=None, **params_to_save):
    config = deepcopy(config)
    config["epochs"] = epochs
    cp_dir = os.path.abspath(cp_dir)
    if args_in_name is not None:
        for i in args_in_name:
            assert i in config
            cp_dir += "-%s_%s" % (i, str(config[i]))

    data = {"config": config}
    data.update(params_to_save)

    filename = os.path.join(cp_dir, f"epochs_{epochs}.pt")
    if os.path.exists(filename):
        print("\x1b[1;33mWarning: Overwriting %s!\x1b[0m" % (filename))

    pathlib.Path(cp_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, filename)


def load_checkpoint(cp_dir, epochs=None, device=None):
    filename = search_cp_file_name(cp_dir, epochs)
    if filename is None:
        raise FileNotFoundError(f"Cannont find the checkpoint in {cp_dir}!")
    if device is not None:
        device = torch.device(device)
    data = torch.load(filename, map_location=device)
    return data


def test_checkpoint(cp_dir, epochs=None, run_epochs=10, **env_kwargs):
    device = get_device()
    model = load_checkpoint(cp_dir, epochs, device)
    cfg = model._cfg
    print(json.dumps(cfg, sort_keys=True, indent=4, default=str))

    cfg["env_kwargs"]["render_mode"] = "human"
    cfg["env_kwargs"].update(env_kwargs)
    cfg["epochs"] = run_epochs * cfg["steps_per_epoch"]
    cfg["steps_per_epoch"] = 1
    cfg["init_stats_param"] = model._env.state_dict()
    if cfg["max_steps_per_traj"] < 0:
        cfg["max_steps_per_traj"] = 1000

    model._init_env()
    model._init_collector()

    for data in model._collector:
        if data["next"]["done"] or data["next"]["truncated"]:
            print(
                f"step_count: %d, traj_reward: %lf"
                % (data["next"]["step_count"].cpu().item(), data["episode_reward"].cpu().item())
            )
        if keyboard.is_pressed("Esc"):
            break
    plt.close(model._fig)
