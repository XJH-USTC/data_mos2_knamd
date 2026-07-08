import numpy as np 
import os
from glob import glob
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


vb_file = sorted(glob('vb_*'))
cb_file = sorted(glob('cb_*'))
print(vb_file)
spin_vb = np.load('vbm_spin150.npy').T
spin_cb = np.load('cbm_spin150.npy').T
print(spin_vb.shape)
print(spin_cb.shape)
vb_pop = []
cb_pop = []

for i in range(len(vb_file)):
    vb_pop.append(np.load(vb_file[i]))
    cb_pop.append(np.load(cb_file[i]))

namdtime = vb_pop[0].shape[0]
namdtime_array = np.arange(1,namdtime+1)
figsize_x = 4.8
figsize_y = 3.2 # in inches
fig, ax = plt.subplots()
fig.set_size_inches(figsize_x, figsize_y)
ax.grid(True, 
        linestyle='--',
        linewidth=0.5,
        color='#808080',
        alpha=0.7,
        which='both')  # 同时显示主次网格
# mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams.update({
    'axes.unicode_minus': False,
    'axes.labelsize': 12
})
band_labels = ['0','LA','TO2','TA','LO1','LO2']
#band_labels = ['0','LA','TO2','TA','LO1']
for i in range(len(vb_pop)):
    vb_i = vb_pop[i]
    cb_i = cb_pop[i]
    print(vb_i.shape)
    # if i ==0:
        # print(vb_i)
        # print(cb_i)
    vb_i = vb_i @ spin_vb
    print(vb_i.shape)
    cb_i = cb_i @ spin_cb
    # if i ==0:
        # print(vb_i)
        # print(cb_i)
    # print(vb_i.T.shape)
    a = vb_i + cb_i
    print(a.shape)
    ax.plot(namdtime_array, a, 
            linewidth=1.5,
            # marker='o' if len(k_indices)==1 else None,
            # markersize=4,
            label=band_labels[i])
# for i in range(pop_index.shape[0]):
#     for j in range(pop_index.shape[1]):
#         index = pop_index[i, j]
#         ax.plot(shp[:, 0], shp[:, index + 2], label='%s_%s' % (band_labels[i],k_labels[j]))
ax.set_xlim(0,namdtime)
# ax.set_ylim(0,1.0)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('\u039C$_{b}$')
handles, labels = ax.get_legend_handles_labels()
ncol = 3 if len(labels) > 6 else 2  # 根据标签数量自动调整列数
ax.legend(handles, labels, 
            fontsize=10,
            ncol=ncol,
            frameon=True,
            shadow=True,
            loc='best')
# ax.legend(fontsize='small',ncol=2)
# plt.yscale('log')
# ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
plt.tight_layout()
plt.savefig('ub.png', dpi=300)
plt.close(fig)
