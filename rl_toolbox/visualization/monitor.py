from typing import Iterable

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.axes._axes import Axes
import numpy as np

from rl_toolbox.utils.backend import check_notebook

in_notebook = check_notebook()
if in_notebook:
    from IPython import display

mpl.rcParams.update({"font.size": 15})


def plot_value(ax: Axes, epoch: int, values: Iterable, value_name=None, handle=None):
    """
    In notebook, use the following code to avoid conflict between `tqdm` and `figure`.

    >>> handle = display.display(fig, display_id=True)
    >>> plot_loss(ax, epoch, losses, handle)

    If the background of the figure is transparent, which may be caused by tqdm, try:

    >>> fig.patch.set_facecolor('white')

    or:
    >>> with plt.style.context('default'):
    >>>     fig = plt.figure()
    """
    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.clear()

    epochs = np.arange(epoch + 1 - len(values), epoch + 1, dtype=np.int32)
    ax.plot(epochs, values, lw=2.5)
    ax.set_xlabel("epoch")
    if value_name is not None:
        ax.set_ylabel(value_name)
    if not in_notebook:
        plt.pause(0.001)
    else:
        if handle is not None:
            handle.update(fig)
        else:
            display.display(fig)
            display.clear_output(wait=True)


def plot_loss(ax: Axes, epoch: int, losses: Iterable, handle=None):
    return plot_value(ax, epoch, losses, "loss", handle)
