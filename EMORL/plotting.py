import matplotlib.pyplot as plt
import os
import sys

import numpy as np


def plot_perf_uniq(perf_and_uniqueness, selected, new_pop, path):
    plt.clf()

    selected_set = set(selected)
    all_indexes = set(range(len(perf_and_uniqueness[0])))
    not_selected = all_indexes-selected_set

    cases = {'new selected' : set(),
    'old selected' : set(),
    'new discarded' : set(),
    'old discarded' : set(),
    }
    for x in selected:
        if x < new_pop.size:
            cases['old selected'].add(x)
        else:
            cases['new selected'].add(x)
    for x in not_selected:
        if x < new_pop.size:
            cases['old discarded'].add(x)
        else:
            cases['new discarded'].add(x)

    plt.style.use(['science', 'scatter', 'grid'])

    print(selected)
    for case, indexes in cases.items():
        print(case, indexes)
        plt.scatter(perf_and_uniqueness[1, list(indexes)], perf_and_uniqueness[0, list(indexes)], label=case, marker='v')


    plt.ylabel(r'$\zeta_{perf}(\pi)$')
    plt.xlabel(r'$\zeta_{nov}(\pi)$')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-51, 51)
    plt.legend()
    plt.draw()

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            print(exc)
    plt.savefig(path+'scatter.png')

    plt.clf()


def plot_stats(population, path):
    plt.style.use(['science', 'ieee'])
    plt.clf()

    #x = np.empty((population.size,), dtype=np.float32)
    index = 0
    r = 3
    c = 5

    dx, dy = 1, 1
    h, w = plt.figaspect(float(dy * r) / float(dx * c))

    fig, plots = plt.subplots(r, c, figsize=(h*4, w*4))



    for hyperparameter in ['experience', 'learning']:
        for name in population.stats['hyperparameter'][hyperparameter][0][0]._variables:
            row = index // c
            col = index % c
            index += 1
            plot = plots[row][col]

            means = []
            mins = []
            maxes = []
            for generation in population.stats['hyperparameter'][hyperparameter]:
                x = [None] * population.size
                for i in range(population.size):
                    x[i] = generation[i][name]
                means.append(np.mean(x))
                mins.append(np.min(x))
                maxes.append(np.max(x))


            plot.plot(np.arange(len(means)), means, label='Mean')
            plot.plot(np.arange(len(means)), mins, label='Min')
            plot.plot(np.arange(len(means)), maxes, label='Max')
            plot.set_ylabel('%s' % name.capitalize().replace('_', ' '))
            plot.set_xlabel('Iterations')
            plot.set_yscale('log')
            plot.legend()

    for name in ['entropy', 'performance']:
        means = []
        mins = []
        maxes = []

        row = index // c
        col = index % c
        index += 1
        plot = plots[row][col]

        for generation in population.stats[name]:
            means.append(np.nanmean(generation))
            mins.append(np.nanmin(generation))
            maxes.append(np.nanmax(generation))
        plot.plot(np.arange(len(means)), means, label='Mean')
        plot.plot(np.arange(len(means)), mins, label='Min')
        plot.plot(np.arange(len(means)), maxes, label='Max')
        plot.set_ylabel('%s' % name.capitalize().replace('_', ' '))
        plot.set_xlabel(r'Iterations')
        plot.legend()

    row = index // c
    col = index % c
    index += 1
    plot = plots[row][col]


    plot.plot(np.arange(len(population.stats['diversity'])), population.stats['diversity'], label='Mean')
    plot.set_ylabel('Diversity')
    plot.set_xlabel('Iterations')

    print('ok')
    fig.savefig(path + 'stats.png')




    