'''
Description:
    Utils for plotting.
'''
import matplotlib.pyplot as plt
import numpy as np


def linearSegmentCMap(num_colors, cmap_name="viridis"):
    cm = plt.get_cmap(cmap_name)
    color_list = [cm(i//3*3.0/num_colors) for i in range(num_colors)]
    return color_list


def _removeTopRightBorders(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def _removeAllBorders(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

if __name__ == '__main__':
    c_l = linearSegmentCMap(num_colors=40, cmap_name="viridis")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(40):
        lines = ax.plot(np.arange(10) * (i + 1))
        lines[0].set_color(c_l[i])
        lines[0].set_linewidth(i % 3 + 1)
    plt.show()