import ast
import numpy as np
import hypertools as hyp
import matplotlib.pyplot as plt




plt.rcParams.update({"font.size":8})
plt.rcParams["font.family"] = "Times New Roman NN"
plt.rcParams["font.weight"] = "normal"
plt.style.use(['science', 'scatter', 'grid'])

datas = []
colors = ['Blues', 'Reds']
#include_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
include_index = [0, 1, 2, 3, 4, 5, 6]
d2_datas = []
d3_datas = []
targets = [['emogi1.txt'], ['emorl1.txt', 'emorl2.txt']]
labels = ['EMOGI-MOP3', 'EMORL-DvD']
path = 'stats_compare/tennis1/'

mins = None
maxes = None

for t in targets:
    if isinstance(t, list):
        d = []
        for tt in t:
            with open(path+tt, 'r') as f:
                x = ast.literal_eval(f.read())

            data = np.zeros((len(x), len(x[list(x.keys())[0]])))
            for i, xx in enumerate(x):
                data[i, :] = list(x[xx].values())
            d.append(data)
        data = np.concatenate(d)
    else:
        with open(path + t, 'r') as f:
            x = ast.literal_eval(f.read())

        data = np.zeros((len(x), len(x[1])))
        for i, xx in enumerate(x):
            data[i,:] = list(x[xx].values())
    data = data[:, include_index]

    #data -= np.min(data, axis=0)[np.newaxis]
    #data = data / np.max(data, axis=0)[np.newaxis]#hyp.normalize(data, normalize='within')
    datas.append(data)

for data in datas:
    if mins is None:
        mins = np.min(data, axis=0)
        maxes = np.max(data, axis=0)
    else:
        mins = np.minimum(mins, np.min(data, axis=0))
        maxes = np.maximum(maxes, np.max(data, axis=0))

#means = np.mean(np.concatenate(datas), axis=0)
#std = np.std(np.concatenate(datas), axis=0)

for i, data in enumerate(datas):
    #data = (data - means) / std
    data = (data - mins) / (maxes-mins)
    datas[i] = data
    d2 = hyp.reduce(data, ndims=2)
    d2_datas.append(d2)

for data in datas:
    d3 = hyp.reduce(data, ndims=3)
    d3_datas.append(d3)


for i, data in enumerate(d2_datas):
    plt.scatter(data[:, 0], data[:, 1], label=labels[i] ,marker='v')

#plt.xticks([])
#plt.yticks([])
#plt.title('Population repartition in the behavior space')
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)


plt.legend()
plt.draw()
plt.savefig(path+'2d.png')
plt.clf()


for i, data in enumerate(datas):
    mobility = (data[:, 4]**2 + data[:, 5]**2)**0.5 / np.sqrt(2)
    efficiency = 1-data[:, 6]
    #x = data[:, 2]
    #let = data[:, 3]
    #x = data[:, 4]
    #y = data[:, 7]
    plt.scatter(mobility, efficiency, label=labels[i] ,marker='v')
plt.legend()
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xticks(np.linspace(0, 1, 5))
plt.yticks(np.linspace(0, 1, 5))
plt.ylabel(r'Mobility')
plt.xlabel(r'Aim')
plt.draw()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path+'mob_eff.png', dpi=500)
plt.clf()



fig = plt.figure()
axd3 = fig.add_subplot(projection='3d')

#axd3.view_init(20, 32)
#axd3.set_xlabel('Aggressiveness')
#axd3.set_ylabel('Defensiveness')
#axd3.set_zlabel('Mobility')
for i, data in enumerate(d3_datas):
    X, Y, Z = np.rollaxis(data, 1)
    axd3.scatter3D(X, Y, Z, c=Z, label=labels[i], linewidth=2., cmap=colors[i])

    #axd3.set_title("Population repartition in the behavior space")
plt.legend()
fig.savefig(path+'3d.png')

#hyp.plot(data, fmt='.', ndims=3, save_path='hyp.png')
#hyp.plot(data[:, [2,3]], fmt='.', ndims=2, title='Mobility in function of aggressiveness', save_path='mob_agg.png')
#hyp.plot(data[:, [2,3,4]], fmt='.', ndims=3, title='Mobility in function of aggressiveness and defensiveness', save_path='mob_agg_def.png')
