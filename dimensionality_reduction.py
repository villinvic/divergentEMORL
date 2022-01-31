import ast
import numpy as np
import hypertools as hyp
import matplotlib.pyplot as plt

plt.style.use(['science', 'scatter'])

datas = []
d2_datas = []
d3_datas = []
targets = ['emogi1.txt', 'emorl1.txt']
path = 'stats_compare/boxing/'
for t in targets:
    with open(path+t, 'r') as f:
        x = ast.literal_eval(f.read())

    data = np.zeros((len(x), len(x[1])))
    for i, xx in enumerate(x):
        data[i,:] = list(x[xx].values())
    data = hyp.normalize(data)
    datas.append(data)

for data in datas:
    d2 = hyp.reduce(data, ndims=2)
    d2_datas.append(d2)

for data in datas:
    d3 = hyp.reduce(data, ndims=3)
    d3_datas.append(d3)


for data in d2_datas:
    plt.scatter(data[:, 0], data[:, 1], marker='v')

plt.title('Population repartition in the behavior space')
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)
plt.legend()
plt.draw()
plt.savefig(path+'2d.png')
plt.clf()
fig = plt.figure()
axd3 = fig.add_subplot(projection='3d')
for data in d3_datas:


    axd3.scatter(data[:, 0], data[:, 1], data[:, 2])
    axd3.set_title("3d behavior space")
    axd3.set_xlabel("l'axe des x")
    axd3.set_ylabel("l'axe des y")
    axd3.set_zlabel("l'axe des z")
    fig.savefig(path+'3d.png')

#hyp.plot(data, fmt='.', ndims=3, save_path='hyp.png')
#hyp.plot(data[:, [2,3]], fmt='.', ndims=2, title='Mobility in function of aggressiveness', save_path='mob_agg.png')
#hyp.plot(data[:, [2,3,4]], fmt='.', ndims=3, title='Mobility in function of aggressiveness and defensiveness', save_path='mob_agg_def.png')
