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

from collections import deque
import pprint
import numpy as np
import nnkit as nn
import json
import gym
import dqn
import os
import plot

pp = pprint.PrettyPrinter(indent=2)

envPrintFreq = 50
envDisplayFreq = None

for envName, solveCriteria in [
    ('CartPole-v0', (195, 100)),
    ('LunarLander-v2', (200, 100)),
    ('MountainCar-v0', (-110, 100))
]:
    env = gym.make(envName)

    if not os.path.exists(envName):
        os.makedirs(envName)

    for n, (steps, exploreSchedule) in enumerate([
        (400000, (1., 0.1, 400000 * 0.1)),
        (500000, (1., 0.1, 500000 * 0.1)),
        (600000, (1., 0.1, 600000 * 0.1)),
        (400000, (1., 0.1, 400000 * 0.5)),
        (500000, (1., 0.1, 500000 * 0.5)),
        (600000, (1., 0.1, 600000 * 0.5)),
        (400000, (1., 0.02, 400000 * 0.1)),
        (500000, (1., 0.02, 500000 * 0.1)),
        (600000, (1., 0.02, 600000 * 0.1)),
        (400000, (1., 0.02, 400000 * 0.5)),
        (500000, (1., 0.02, 500000 * 0.5)),
        (600000, (1., 0.02, 600000 * 0.5)),
    ]):

        modelPath = envName + '/' + envName + '-m' + str(n)
        plotPath = modelPath + '.training'
        statsPath = plotPath + '.json'

        rewards = deque(maxlen=solveCriteria[1])
        avgRewards = float('-inf')
        solved = False
        stats = []

        settings = dict(
            env=env,
            seed=77,
            hiddenSize=512,
            discount=0.99,
            steps=steps,
            learnRate=1e-3,
            exploreSchedule=exploreSchedule,
            replayMin=2000,
            replayMax=50000,
            replayBatch=32,
            targetUpdateFreq=500,
            loss=nn.L2Loss,
            renderFreq=envDisplayFreq
        )

        settingsInfo = settings.copy()
        settingsInfo['env'] = envName
        settingsInfo['loss'] = settingsInfo['loss'].__name__
        stats.append(settingsInfo)
        stats.append({'solve criteria': solveCriteria})
        stats.append([])

        print('++ TRAINING: {} | solve criteria: {} ++'.format(envName, solveCriteria))
        print('Training Settings:')
        pp.pprint(settingsInfo)

        for e, t, eps, l, r, Q in dqn.train(**settings):
            rewards.append(r)
            avg = np.mean(rewards)
            stats[-1].append((t, eps, l, r, avg))
            alert = None

            if avg > avgRewards:
                avgRewards = avg
                alert = '*R*'

            if alert or e % envPrintFreq is 0:
                print('[TRAINING ({:.2%})] e:{} | t:{} | eps:{:,.3f} | l:{:,.3f} | r:{:,.3f} | avg:{:,.3f} | {}'.format(
                    (t + 1) / steps, e, t, eps, l, r, avg, alert
                ))

        nn.save(Q.topology, modelPath)
        with open(statsPath, 'wt') as file:
            json.dump(stats, file)

        plot.plotStats(statsPath, savePath=plotPath)

    env.close()
