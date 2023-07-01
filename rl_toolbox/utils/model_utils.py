from copy import deepcopy
import os
import pathlib
import re

import json
import keyboard
import numpy as np
from tqdm.auto import trange

import gymnasium as gym
import torch
from torch import nn
from torch.optim import Optimizer
import safetensors # TODO

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

    for key in params_to_save:
        if isinstance(params_to_save[key], (nn.Module, Optimizer)):
            params_to_save[key] = params_to_save[key].state_dict()

    data = {"config": config}
    data.update(params_to_save)

    filename = os.path.join(cp_dir, f"epochs_{epochs}.pt")
    if os.path.exists(filename):
        print("\x1b[1;33mWarning: Overwriting %s!\x1b[0m" % (filename))

    pathlib.Path(cp_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, filename)


def load_checkpoint(cp_dir, epochs=None, device=None):
    filename = search_cp_file_name(cp_dir, epochs)
    if device is not None:
        device = torch.device(device)
    data = torch.load(filename, map_location=device)
    return data


def test_checkpoint(cp_dir, epochs=None, run_times=10, **env_config):
    device = get_device()
    data = load_checkpoint(cp_dir, epochs, device)
    config = data["config"]

    print(json.dumps(config, sort_keys=True, indent=4, default=str))

    config["env_config"].update(env_config)

    env = config["env"]
    continuous = config["continuous"]

    if isinstance(env, str):
        env = gym.make(
            env,
            max_episode_steps=config["max_steps_per_traj"],
            render_mode="human",
            **config["env_config"],
        )
    else:
        env.unwrapped.render_mode = "human"

    obs_space_size = env.observation_space.shape[0]

    if continuous:
        action_space_size = env.action_space.shape[0]
    else:
        action_space_size = env.action_space.n

    model = data["__class__"].get_model(obs_space_size, action_space_size, config, data)

    for _ in trange(run_times):
        obs, info = env.reset()
        done = False
        while not done:
            if keyboard.is_pressed("Esc"):
                return
            action = model.select_action(as_tensor32(obs).to(device))
            if isinstance(action, tuple):
                action = action[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    # env.close()
    return info
