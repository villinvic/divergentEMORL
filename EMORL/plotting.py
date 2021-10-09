import matplotlib.pyplot as plt


def plot_stats(perf_and_uniqueness, selected, new_pop, path):
    selected_set = set(selected)
    all_indexes = set(range(len(perf_and_uniqueness)))
    not_selected = all_indexes-selected_set

    cases = {'new_selected' : set(),
    'old_selected' : set(),
    'new_discarded' : set(),
    'old_discarded' : set(),
    }
    for x in selected:
        if x < new_pop.size:
            cases['old_selected'].add(x)
        else:
            cases['new_selected'].add(x)
    for x in not_selected:
        if x < new_pop.size:
            cases['old_discarded'].add(x)
        else:
            cases['new_discarded'].add(x)

    plt.style.use(['science', 'scatter', 'grid'])

    for case, indexes in cases.items():
        plt.scatter(perf_and_uniqueness[1, indexes], perf_and_uniqueness[0, indexes], label=case, marker='v')


    plt.xlabel('r$\zeta_{perf}(\pi)$')
    plt.ylabel('r$\zeta_{kl}(\pi)$')
    plt.legend()
    plt.draw()
    plt.savefig(path+'scatter.png')