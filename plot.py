# The MIT License (MIT)
#
# Copyright (c) 2018 Federico Saldarini
# https://www.linkedin.com/in/federicosaldarini
# https://github.com/saldavonschwartz
# https://0xfede.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def mark(eRange, xy, ax, ax1, ax2, color, text=''):
    size, alpha = 120, 0.5

    ax.scatter([xy[0]], [xy[1]], s=size, alpha=alpha, color=color)
    ax.annotate(
        s='{:,.2f} {}'.format(xy[1], text),
        xy=(xy[0] + 0.005, xy[1] + 0.005), fontsize=8, horizontalalignment='right'
    )

    ax1.set_xticks(list(ax1.get_xticks()) + [xy[0]])
    ax2.set_xticks(list(ax2.get_xticks()) + [xy[0]])
    ax1.set_xlim(1, eRange)
    ax2.set_xlim(1, eRange)
    ax1.axvline(x=xy[0], linewidth=0.5, alpha=0.4, color=(0,0,0), drawstyle='steps-pre')
    plt.setp(ax1.get_xticklabels(), rotation=30, alpha=0.8, fontsize=6, horizontalalignment='right')


def markSolve(ax, solveCriteriaValue):
    return ax.axhline(
        label='solve criteria: {}'.format(solveCriteriaValue), y=solveCriteriaValue,
        linewidth=1., alpha=1, color=(.3, .7, .7), drawstyle='steps-pre', linestyle='--')


def plotStats(path, savePath):
    """Plot stats for a training session from a JSON file, optionally saving the plot as an SVG"""
    with open(path, 'rt') as file:
        data = json.load(file)

    epsSchedule = data[0]['epsSchedule']
    steps = data[0]['steps']
    solveCriteria = data[1]['solve criteria']
    stats = data[2]

    eRange = list(range(len(stats)))
    l = np.array([s[2] for s in stats])
    l /= np.max(l)
    r = [s[3] for s in stats]

    eps = [s[1] for s in stats]
    avg = [s[4] for s in stats]
    lLow = [np.argmin(l), np.min(l)]
    lHigh = [np.argmax(l), np.max(l)]
    highR = [np.argmax(r), np.max(r)]
    highAvg = [np.argmax(avg), np.max(avg)]

    figure = plt.figure(figsize=(9, 6))
    ax = figure.add_subplot(1, 1, 1)
    ax.tick_params(labelsize=8)
    ax.set_xlim(eRange[0], eRange[-1])
    ax.set_xlabel('episodes (after replay buffer size >= min)', fontsize=8)
    ax.set_ylabel('loss (normalized to [0,1]) and ' + r'$\epsilon$', fontsize=8)
    ax.set_ylim(0, 1)
    ax.autoscale(enable=True, axis='y', tight=None)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    p1 = ax.plot(
        eRange, l,
        label='avg loss (over 1 episode)', color=(237/255, 102/255, 93/255), alpha=0.7
    )[0]

    p2 = ax.plot(
        eRange, eps,
        label=r'$\epsilon$', color=(109 / 255, 204 / 255, 218 / 255)
    )[0]

    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(-200, -80)
    ax2.autoscale(enable=True, axis='y', tight=None)
    ax2.set_ylabel('rewards', fontsize=8)

    p3 = ax2.plot(
        eRange, avg,
        label='avg rewards (over {} episodes)'.format(solveCriteria[1]), color=(255/255, 158/255, 74/255)
    )[0]

    # mark(eRange[-1], lLow, ax, ax, ax2, (237/255, 102/255, 93/255), 'lowest loss')
    # mark(eRange[-1], lHigh, ax, ax, ax2, (237 / 255, 102 / 255, 93 / 255), 'highest loss')
    # mark(eRange[-1], highR, ax2, ax, ax2, (109/255, 204/255, 218/255), 'highest reward')
    mark(eRange[-1], highAvg, ax2, ax, ax2, (255/255, 158/255, 74/255), 'best avg')
    solvedInTraining = np.where(np.array(avg) >= solveCriteria[0])[0]

    p4 = markSolve(ax2, solveCriteria[0])

    ax.legend(
        [p1, p2, p3, p4], [p1.get_label(), p2.get_label(), p3.get_label(), p4.get_label()],
        loc='lower center', frameon=False, bbox_to_anchor=(0, 1, 1., 1), ncol=4
    )

    plt.title('DQN Training Stats:\n\n{} | steps: {} | episodes: {} | {}-schedule: {}\n\n'.format(
        os.path.split(path)[-1].split('.')[0],
        steps, len(eRange), r'$\epsilon$', epsSchedule
    ), fontsize=10)

    plt.tight_layout()

    if savePath is not None:
        plt.savefig(
            savePath + '.svg',
            format='svg', dpi=1200, transparent=True
        )
    else:
        plt.show(block=False)
        return plt


