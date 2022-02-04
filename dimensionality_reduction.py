import ast
import numpy as np
import hypertools as hyp
import matplotlib.pyplot as plt




plt.rcParams.update({"font.size":8})
plt.rcParams["font.family"] = "Times New Roman NN"
plt.rcParams["font.weight"] = "normal"

datas = []
colors = ['Blues', 'Oranges']
include_index = [0, 1, 2, 3, 4] #
d2_datas = []
d3_datas = []
targets = [['emogi1.txt', 'emogi2.txt'], ['emorl1.txt', 'emorl2.txt']]
labels = ['EMOGI-MOP1', 'EMORL-DvD']
path = 'stats_compare/boxing/'

for t in targets:
    if isinstance(t, list):
        d = []
        for tt in t:
            with open(path+tt, 'r') as f:
                x = ast.literal_eval(f.read())

            data = np.zeros((len(x), len(x[1])))
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

fig = plt.figure()
axd3 = fig.add_subplot(projection='3d')

axd3.view_init(20, 32)
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
