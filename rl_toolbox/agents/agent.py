from abc import ABC, abstractmethod
import os

from matplotlib import pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from rl_toolbox.utils.backend import check_notebook
from rl_toolbox.utils.cp_utils import load_checkpoint, save_checkopoint
from rl_toolbox.visualization.monitor import plot_value

in_notebook = check_notebook()
if in_notebook:
    from IPython import display


class Agent(ABC):
    def __init__(
        self, env_name, cp_dir, save_buf, vis_value_names, saved_attr_names, **cfg
    ) -> None:
        if cp_dir is not None and os.path.exists(cp_dir):
            raise FileExistsError(f"Directory {cp_dir} already exists!")
        self._cp_dir = cp_dir
        self._env_name = env_name
        self._save_buf = save_buf
        self._cfg = self._updated_cfg(cfg)

        self._curr_epoch = 0
        self._vis_value_names = vis_value_names
        self._saved_attr_names = saved_attr_names

    def learn(self):
        self._init_all()

        pbar = tqdm(
            total=self._cfg["epochs"] - self._curr_epoch, initial=self._curr_epoch
        )
        while self._curr_epoch < self._cfg["epochs"]:
            self._curr_epoch += 1

            log = self.one_epoch()
            self._process_log(log)
            self._update_vis()

            if self._need_to_save():
                self._save_checkpoint()

            pbar.update()

    @abstractmethod
    def one_epoch(self):
        raise NotImplementedError()

    @classmethod
    def from_checkpoint(
        cls, cp_dir, epochs, render_mode=None, human_render_hook=None, **env_kwargs
    ):
        return load_checkpoint(
            cls, cp_dir, epochs, render_mode, human_render_hook, **env_kwargs
        )

    @abstractmethod
    def _init_all(self):
        raise NotImplementedError()

    def _init_vis(self):
        if self._vis_value_names is None:
            return

        n = len(self._vis_value_names)
        if hasattr(self, "_fig"):
            plt.close(self._fig)
        fig = plt.figure(figsize=[5.5 * n, 4])
        self._fig = fig
        self._axes = dict()
        for i, name in enumerate(self._vis_value_names):
            self._axes[name] = fig.add_subplot(1, n, i + 1)
        self._display_handle = None
        if in_notebook:
            self._display_handle = display.display(fig, display_id=True)

    def _update_vis(self):
        if self._vis_value_names is None:
            return
        if self._log is None:
            return

        for name in self._vis_value_names:
            plot_value(
                self._axes[name],
                self._curr_epoch,
                self._log[self._log[name].notna()][name].to_numpy(),
                name,
                self._display_handle,
            )

    def _process_log(self, log: dict):
        # lazy init
        if not hasattr(self, "_log"):
            self._log = pd.DataFrame(columns=log.keys())

        self._log = self._log.append(log, ignore_index=True)

    def _need_to_save(self):
        save_cp_freq = self._cfg["save_cp_freq"]
        return (
            self._cp_dir is not None
            and save_cp_freq is not None
            and self._curr_epoch % save_cp_freq == 0
        )

    def _save_checkpoint(self):
        save_checkopoint(self, ["_" + name for name in self._saved_attr_names])

    def _updated_cfg(self, cfg: dict):
        dflt_cfg = self._get_dflt_cfg()
        for key in cfg.keys():
            if key not in dflt_cfg:
                raise KeyError(f"Invalid key {key} in cfg")

        dflt_cfg.update(cfg)
        cfg = dflt_cfg

        num_batch = int(cfg["steps_per_epoch"] // cfg["batch_size"])
        cfg["num_batch"] = num_batch
        steps_per_epoch = num_batch * cfg["batch_size"]
        if steps_per_epoch != cfg["steps_per_epoch"]:
            print(
                f"\x1b[1;33mWarning: `steps_per_epoch` is truncated to {steps_per_epoch}!\x1b[0m"
            )
            cfg["steps_per_epoch"] = steps_per_epoch
        if cfg["steps_per_epoch"] < cfg["batch_size"]:
            raise ValueError(
                "`steps_per_epoch` (%d) < `batch_size` (%d)!"
                % (cfg["steps_per_epoch"], cfg["batch_size"])
            )
        return cfg
