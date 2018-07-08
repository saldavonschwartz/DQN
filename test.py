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

import nnkit as nn
import json
import gym
import dqn
import os

envDisplayFreq = 25
envRecordFreq = None
runs = 3

for envName, solveCriteria in [
    ('CartPole-v0', (195, 100)),
    ('LunarLander-v2', (200, 100)),
    ('MountainCar-v0', (-110, 100))
]:
    models = [m.replace('.model.gz', '') for m in os.listdir(envName) if '.model' in m]
    statsPath = envName + '/' + envName + '.testing.json'

    envBase = gym.make(envName)
    stats = {}

    for modelName in models:
        modelPath = envName + '/' + modelName
        stats[modelName] = {'runs': []}
        Q = nn.FFN(*nn.load(modelPath))

        print('++ TESTING: {} | solve criteria: {} ++'.format(modelName, solveCriteria))
        avgReward = 0

        if envRecordFreq is not None:
            videoPath = envName + '/video/' + modelName
            env = gym.wrappers.Monitor(
                envBase, videoPath, force=True, video_callable=lambda e: z is 0 and e % envRecordFreq is 0
            )
        else:
            env = envBase

        for z in range(runs):
            for e, r, avg, solved in dqn.test(env, Q, solveCriteria, envDisplayFreq):
                if e % envDisplayFreq is 0:
                    print('e: {}'.format(e))

            print('run: {} | avg: {:,.3f} | solved: {}'.format(z, avg, solved))
            stats[modelName]['runs'].append((avg, bool(solved)))
            avgReward += avg

        avgReward /= runs
        solved = bool(avgReward >= solveCriteria[0])
        print('total avg reward: {:,.3f} | solved: {}'.format(avgReward, solved))
        stats[modelName]['score'] = avgReward, solved

    with open(statsPath, 'wt') as file:
        json.dump(stats, file)

    env.close()





