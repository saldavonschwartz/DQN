import nnkit as nn
import json
import gym
import dqn
import os
import plot



# statsPath = envName + '/' + envName + '.testing.json'
# plot.plotStats(statsPath, savePath=plotPath)

# Q2. For each problem, how many successful.
# Q3. For each problem, diff between common and best.
# Q4. For best and common, std dev between runs.
# Q5. For all models, mean and std dev.
#

from collections import OrderedDict

stats = {}
envNames = ['CartPole-v0', 'LunarLander-v2', 'MountainCar-v0']

for envName in envNames:
    with open(envName + '/' + envName + '.testing.json', 'rt') as file:
        data = json.load(file)

    ordered = OrderedDict()
    keys = sorted(data, key=lambda k: data[k]['score'][0], reverse=True)
    for k in keys:
        ordered[k] = data[k]

    stats[envName] = ordered

for env in stats:
    print('------------ {} -------------'.format(env))
    winners = 0
    best = 0
    general = 0

    for m in stats[env]:
        mdata = stats[env][m]
        if mdata['score'][1] is True:
            winners += 1

            if not best:
                best = mdata['score'][0]

        scores = ['{:,.3f} | '.format(s) for s in [s[0] for s in mdata['runs']]] + ['{:,.3f}'.format(mdata['score'][0])]
        row = m.split('-')[-1][1:] + ' | '

        if m.split('-')[-1][1:] == '10':
            general = mdata['score'][0]

        # print(row, *scores)

    # print('success model ratio: {}/{} = {:,.3f}'.format(winners, len(stats[envName]), winners / len(stats[envName])))
    print('Best / General Model Improvement: {:,.3f}/{:,.3f} = {:,.3f}'.format(best, general, abs(best-general)/max(abs(best), abs(general))))


print('\nQ1. Common models to all problems:')

common = None
for envName in envNames:
    s = set([k.split(envName)[-1] for k in stats[envName] if stats[envName][k]['score'][1] is True])
    common = s if not common else common & s

commonScores = []
for m in common:
    commonScores.append([(e, stats[e][e + m]['score'][0], stats[e][e + m]['runs']) for e in envNames])
    print('common:', m)
    for c in commonScores[-1]:
        print(c[:2])
        print(c[2:])
        print('std dev')
        print('\n')


bestScores = []
for envName in envNames:
    stat = stats[envName]
    k = max(stat, key=lambda k: stat[k]['score'][0])
    bestScores.append((k, stat[k]['score'][0], stat[k]['runs']))

print('\nQ2. Best scores:')
for b in bestScores:
    print(b[:2])
    print(b[2:])
    print('std dev')
    print('\n')

print('\nQ3. Diff best vs common:')
for i in range(3):
    b = bestScores[i][1]
    c = commonScores[-1][i][1]
    print('b:{:,.2f} vs c:{:,.2f} | {:,.2f} | {:,.2f}'.format(b, c, b/c, abs(b-c)/max(abs(b), abs(c))))








# m = sorted(m, key=lambda k: data[k]['score'][0], reverse=True)



