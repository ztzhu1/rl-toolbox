import inspect
import json
import os
import pathlib
import re

import keyboard
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import torch

from importlib import import_module


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


def save_checkopoint(agent, saved_attr_names):
    cp_dir = agent._cp_dir
    if os.path.exists(cp_dir):
        raise FileExistsError(f"Directory {cp_dir} already exists!")
    cfg_path = os.path.join(cp_dir, f"config.pt")
    log_path = os.path.join(cp_dir, f"log.csv")
    data_path = os.path.join(cp_dir, f"epochs_{agent._curr_epoch}.pt")

    pathlib.Path(agent._cp_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(cfg_path):
        torch.save(agent._cfg, cfg_path)
    agent._log.to_csv(log_path, index=False)

    class_name = type(agent).__name__
    class_abs_path = inspect.getfile(agent.__class__)
    _, class_filename = os.path.split(class_abs_path)
    class_filename_wo_ext = os.path.splitext(class_filename)[0]
    entry_point = class_filename_wo_ext + "." + class_name
    data = dict(entry_point=entry_point)
    for name in saved_attr_names:
        attr = getattr(agent, name)
        if hasattr(attr, "state_dict"):
            attr = attr.state_dict()
            name += "-state_dict"
        data[name] = attr
    if agent._save_buf:
        data["buf"] = agent._buf
    torch.save(data, data_path)


def load_checkpoint(
    class_, cp_dir, epochs=None, render_mode=None, human_render_hook=None, **env_kwargs
):
    cfg = torch.load(os.path.join(cp_dir, "config.pt"), map_location="cpu")
    log = pd.read_csv(os.path.join(cp_dir, "log.csv"))
    data = torch.load(search_cp_file_name(cp_dir, epochs), map_location="cpu")
    print(json.dumps(cfg, sort_keys=True, indent=4, default=str))
    cfg["init_stats_param"] = data["_env-state_dict"]

    agent = class_(cp_dir=None, env_name=data["_env_name"], **cfg)
    agent._curr_epoch = data["_curr_epoch"]
    agent._vis_value_names = data["_vis_value_names"]
    agent._log = log[:epochs]

    if render_mode is not None:
        # testing mode
        cfg = agent._cfg
        assert render_mode == "human"
        cfg["env_kwargs"].update(env_kwargs)
        cfg["env_kwargs"]["render_mode"] = render_mode
        cfg["epochs"] = 10 * cfg["steps_per_epoch"]
        cfg["steps_per_epoch"] = 1
        if cfg["max_steps_per_traj"] < 0:
            cfg["max_steps_per_traj"] = 1000
        if human_render_hook is not None:
            human_render_hook(cfg)

    agent._init_all()
    agent._update_vis()

    for key in data:
        if key.endswith("-state_dict"):
            agent.__dict__[key[:-11]].load_state_dict(data[key])

    if render_mode != "human" and "buf" in data:
        agent._buf = data["_buf"]

    return agent


def test_checkpoint(cp_dir, epochs=None, **env_kwargs):
    data = torch.load(search_cp_file_name(cp_dir, epochs), map_location="cpu")
    agent_file, agent_class = data["entry_point"].split(".")
    m = import_module(f"rl_toolbox.agents.{agent_file}")
    class_ = getattr(m, agent_class)

    agent = class_.from_checkpoint(cp_dir, epochs, "human", **env_kwargs)

    for data in agent._collector:
        data_next = data["next"]
        if data_next["done"] or data_next["truncated"]:
            print(
                f"step_count: %d, traj_reward: %lf"
                % (
                    data_next["step_count"].cpu().item(),
                    data_next["episode_reward"].cpu().item(),
                )
            )
        if keyboard.is_pressed("Esc"):
            break
    plt.close(agent._fig)
