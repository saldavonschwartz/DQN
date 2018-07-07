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
from copy import deepcopy
import random

import nnkit as nn
import numpy as np
import gym


def train(env, seed, hiddenSize, discount, steps, learnRate, exploreSchedule,  replayMin, replayMax, replayBatch, targetUpdateFreq, loss, renderFreq):
    if seed:
        np.random.seed(seed)

    ins = env.observation_space.shape[0]

    if type(env.action_space) is gym.spaces.box.Box:
        outs = env.action_space.shape[0]
    else:
        outs = env.action_space.n

    Q = nn.FFN(
        (nn.Multiply, nn.rand2(ins, hiddenSize)), (nn.Add, nn.rand2(hiddenSize)), (nn.ReLU,),
        (nn.Multiply, nn.rand2(hiddenSize, outs)), (nn.Add, nn.rand2(outs))
    )

    Qt = deepcopy(Q)
    optimizer = nn.Adam(Q.vars)
    optimizer.learnRate = learnRate
    replay = deque(maxlen=replayMax)
    e, t, l, rewards = 0, 0, float('inf'), 0
    totalSteps = steps + replayMin
    episodeLoss = []
    s0 = env.reset()

    while t < totalSteps:
        k = max(0, min(1, (t-(replayMin-1)) / (exploreSchedule[2]-1)))
        eps = k * exploreSchedule[1] + (1 - k) * exploreSchedule[0]

        if np.random.rand() < eps:
            a0 = env.action_space.sample()
        else:
            a0 = np.argmax(Q(nn.NetVar(s0)))

        s1, r1, done, _ = env.step(a0)
        replay.append((s0, a0, s1, r1, done))
        rewards += r1

        if renderFreq is not None and e % renderFreq is 0:
            env.render()

        if t >= (replayMin - 1):
            samples = np.asarray(random.sample(replay, replayBatch)).T
            S0, A0, S1, R1, D = [np.array(s.tolist()) for s in samples]

            target = R1 + discount * np.max(Qt(nn.NetVar(S1)), axis=1) * ~D
            Y = np.copy(Q(nn.NetVar(S0)))
            Y[range(replayBatch), A0] = target

            l = loss(Q.layers[-1], nn.NetVar(Y))
            episodeLoss.append(l.data.item())
            l.back()

            optimizer.step()

            if ((t - (replayMin - 1)) + 1) % targetUpdateFreq is 0:
                Qt = deepcopy(Q)

        if done:
            if t >= (replayMin - 1):
                yield e, t, eps, np.mean(episodeLoss), rewards, Q
                episodeLoss.clear()

            e, rewards, done = e+1, 0, False
            s0 = env.reset()
        else:
            s0 = s1

        t += 1


def test(env, Q, solvedCriteria, renderFreq):
    episodeRewards = deque(maxlen=solvedCriteria[1])
    e = 0

    while e < solvedCriteria[1]:
        rewards, done = 0, False
        s0 = env.reset()

        while not done:
            a0 = np.argmax(Q(nn.NetVar(s0)))
            s1, r1, done, _ = env.step(a0)
            rewards += r1
            s0 = s1

            if renderFreq is not None and e % renderFreq is 0:
                env.render()

        episodeRewards.append(rewards)
        avgRewards = np.mean(episodeRewards)
        solved = len(episodeRewards) is solvedCriteria[1] and avgRewards >= solvedCriteria[0]
        yield e, rewards, avgRewards, bool(solved)
        e += 1
