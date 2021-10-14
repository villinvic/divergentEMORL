import matplotlib.pyplot as plt
import os


def plot_stats(perf_and_uniqueness, selected, new_pop, path):
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
    plt.xlabel(r'$\zeta_{kl}(\pi)$')
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.draw()

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            print(exc)
    plt.savefig(path+'scatter.png')

    plt.clf()

    