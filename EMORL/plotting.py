import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import os
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


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

    for c in cases:
        cases[c] = [perf_and_uniqueness[1, list(cases[c])], perf_and_uniqueness[0, list(cases[c])]]

    plt.style.use(['science', 'scatter', 'grid'])

    print(selected)
    for case, (div, perf) in cases.items():
        print(case, div)
        plt.scatter(perf, div, label=case, marker='v')


    plt.ylabel(r'$\zeta_{perf}(\pi)$')
    plt.xlabel(r'$\zeta_{nov}(\pi)$')
    #plt.xlim(-0.05, 1.05)
    plt.ylim(-55, 55)
    #plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.draw()

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            print(exc)
    plt.savefig(path+'scatter.png')

    plt.clf()


def plot_evo(pop, path):
    plt.style.use(['ieee', 'science', 'scatter', 'grid'])
    plt.clf()

    perf = np.empty((pop.size,), dtype=np.float32)
    div = np.empty((pop.size,), dtype=np.float32)



    for i, inv in enumerate(pop):
        perf[i] = (inv.performance  + 50 )/100.
        div[i] = inv.div_score

    plt.scatter(div, perf, marker='v')
    plt.ylabel(r'$\zeta_{perf}(\pi)$')
    plt.xlabel(r'$\zeta_{nov}(\pi)$')
    plt.ylim(-0.05, 1.05)
    plt.draw()
    plt.savefig(path + 'evo.png')




def plot_stats(population, path):
    plt.style.use(['science', 'ieee', 'grid'])
    plt.clf()

    #x = np.empty((population.size,), dtype=np.float32)
    index = 0
    r = 3
    c = 5

    dx, dy = 1, 1
    h, w = plt.figaspect(float(dy * r) / float(dx * c))

    #fig, plots = plt.subplots(r, c, figsize=(h*4, w*4))



    for hyperparameter in ['experience', 'learning']:
        for name in population.stats['hyperparameter'][hyperparameter][0][0]._variables:
            row = index // c
            col = index % c
            index += 1
            #plot = plots[row][col]

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

            x = np.arange(len(means))
            xnew = np.linspace(0, len(means), 50)

            spl = make_interp_spline(x, means, k=3)  # type: BSpline
            means = spl(xnew)
            spl = make_interp_spline(x, mins, k=3)  # type: BSpline
            mins = spl(xnew)
            spl = make_interp_spline(x, maxes, k=3)  # type: BSpline
            maxes = spl(xnew)

            plt.plot(xnew, means, label='Mean')
            plt.plot(xnew, mins, label='Min')
            plt.plot(xnew, maxes, label='Max')
            plt.ylabel('%s' % name.capitalize().replace('_', ' '))
            plt.xlabel('Iterations')
            plt.yscale('log')
            plt.legend()
            plt.savefig(path + name + '.png')
            plt.clf()


    for name in ['entropy', 'performance']:
        means = []
        mins = []
        maxes = []

        row = index // c
        col = index % c
        index += 1
        #plot = plots[row][col]

        for generation in population.stats[name]:
            generation = np.array(generation)
            mask = np.isfinite(generation)
            generation[~mask] = np.mean(generation[mask])

            if name=="performance":
                generation=(generation + 50) /100.

            means.append(np.nanmean(generation))
            mins.append(np.nanmin(generation))
            maxes.append(np.nanmax(generation))

        x = np.arange(len(means))
        xnew = np.linspace(0, len(means), 50)

        spl = make_interp_spline(x, means, k=3)  # type: BSpline
        means = spl(xnew)
        spl = make_interp_spline(x, mins, k=3)  # type: BSpline
        mins = spl(xnew)
        spl = make_interp_spline(x, maxes, k=3)  # type: BSpline
        maxes = spl(xnew)

        if name == 'performance':
            name = 'Winrate'

        plt.plot(np.arange(len(means)), means, label='Mean')
        plt.plot(np.arange(len(means)), mins, label='Min')
        plt.plot(np.arange(len(means)), maxes, label='Max')
        plt.ylabel('%s' % name.capitalize().replace('_', ' '))
        plt.yscale('linear')
        plt.xlabel(r'Iterations')
        plt.legend()
        plt.savefig(path + name + '.png')
        plt.clf()

    row = index // c
    col = index % c
    index += 1


    plt.plot(np.arange(len(population.stats['diversity'])), population.stats['diversity'], label='Mean')
    plt.ylabel('Diversity')
    plt.xlabel('Iterations')
    plt.yscale('log')

    print('ok')
    plt.savefig(path + 'diversity.png')



def data_coord2view_coord(p, vlen, pmin, pmax):
    dp = pmax - pmin
    dv = (p - pmin) / dp * vlen
    return dv

def nearest_neighbours(xs, ys, reso, n_neighbours):
    im = np.zeros([reso, reso])
    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]

    xv = data_coord2view_coord(xs, reso, extent[0], extent[1])
    yv = data_coord2view_coord(ys, reso, extent[2], extent[3])
    for x in range(reso):
        for y in range(reso):
            xp = (xv - x)
            yp = (yv - y)

            d = np.sqrt(xp**2 + yp**2)

            im[y][x] = 1 / np.sum(d[np.argpartition(d.ravel(), n_neighbours)[:n_neighbours]])

    im = im ** 0.1

    return im, extent


def heatmap(trajectory, path, name='heatmap', title='Location heatmap'):
    # traj.shape -> (N, 2)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    y = np.max(y) - y
    resolution = 500
    plt.style.use(['science', 'ieee'])
    plt.clf()
    neighbours = 16
    im, extent = nearest_neighbours(x, y, resolution, neighbours)
    plt.imshow(im, origin='lower', cmap=cm.jet)
    #plt.xlim(extent[0], extent[1])
    #plt.ylim(extent[2], extent[3])
    plt.axis('off')

    #plt.title(title)
    plt.savefig(path + name+'.png')
    plt.axis('on')






    