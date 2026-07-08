#!/usr/bin/env python

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
from optparse import OptionParser
from matplotlib.ticker import AutoMinorLocator
import postnamd_250225 as pn
from scipy.optimize import curve_fit
import math
from glob import glob
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg


mpl.rcParams['axes.unicode_minus'] = False
################################################################################
################################################################################

def main():
    fig = plt.figure(
    figsize=(8,10),
    dpi=480,
    )
    ################################################################################
    axes2 = fig.subplot_mosaic(
            [['A','B']],
            gridspec_kw={
                "bottom": 0.02,
                "top": 0.42,
                "left": 0.04,
                "right": 0.95,
                "wspace": 0.2,
                "hspace": 0.1,
            },
            empty_sentinel=-1,
            )
    img_path = ['spin_valley_part3.3.png','spin_valley_part2.4.png']
    graph_label = ['A','B']
    for i in range(2):
        img = mpimg.imread(img_path[i])
        ax_toc = axes2[graph_label[i]]
        ax_toc.imshow(img)
        ax_toc.axis('off')
        ax_toc.set_aspect('equal')
        label = 'dh'[i]
        ax_toc.text(-0.05, 0.98, "{}".format(label),
                    ha="center",
                    va="center",
                    fontsize=14,
                    family='monospace',
                    fontweight='bold',
                    # rotation=45,
                    transform=ax_toc.transAxes,
                    # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
            )        


    axes1 = fig.subplot_mosaic(
        [['A','AA'],['B','BB'],['C','CC']],
        gridspec_kw={
            "bottom": 0.40,
            "top": 0.98,
            "left": 0.08,
            "right": 0.95,
            "wspace": 0.3,
            "hspace": 0.2,
            "height_ratios": [1, 1, 1]
        },
        # height_ratios=[1, 2.0],
        # width_ratios=[1, 1],
    )
    for i,j in enumerate(axes1):
        label = 'aebfcg'[i]
        axes1[j].text(-0.16, 1.00, "{}".format(label),
                    ha="center",
                    va="center",
                    fontsize=14,
                    family='monospace',
                    fontweight='bold',
                    # rotation=45,
                    transform=axes1[j].transAxes,
                    # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
            )
        axes1['A'].set_title(r'$\mathbf{LA@\Gamma\ excitation}$', fontsize=13, pad=12, fontweight='bold')
        axes1['AA'].set_title(r'$\mathbf{TO_2@K\ excitation}$', fontsize=13, pad=12, fontweight='bold')
    vb_file = ['part1/vbm/fssh_pop_sh-0.npy','part2/vbm/fssh_pop_sh-0.npy']
    cb_file = ['part1/cbm/fssh_pop_sh-0.npy','part2/cbm/fssh_pop_sh-0.npy']
    graph_label = ['C','CC']
    for i in range(2):
        plot_ub(axes1[graph_label[i]],vb_file[i],cb_file[i])
    ##
    bassel = np.load('part1/cbm/bassel.npy')+1
    kpts = np.load('part1/cbm/kpts.npy')
    spin_cb = np.load('part1/cbm/cbm_spin150.npy')
    k_indexs = [[0.333333,0.333333,0],[0.666667,0.666667,0]]
    k_labels = ['+K','-K']
    band_indexs = [1,2]
    pop_index = []
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    for i in range(kpts.shape[0]):
        if (periodic_distance(kpts[i], k_indexs[0]))<0.15 and spin_cb[i]< -0.0:
            k1.append(i)
        if (periodic_distance(kpts[i], k_indexs[0]))<0.15 and spin_cb[i]> -0.0:
            k2.append(i)
        if (periodic_distance(kpts[i], k_indexs[1]))<0.15 and spin_cb[i]> -0.0:
            k3.append(i)
        if (periodic_distance(kpts[i], k_indexs[1]))<0.15 and spin_cb[i]< -0.0:
            k4.append(i)
    pop_index=[[k1,k3],[k2,k4]]
    graph_label = ['A','AA']
    shp_e = ['part1/cbm/fssh_e_sh-0.npy','part2/cbm/fssh_e_sh-0.npy']
    shp_pop = ['part1/cbm/fssh_pop_sh-0.npy','part2/cbm/fssh_pop_sh-0.npy']
    for i in range(len(shp_pop)):
        shp_e1 = np.load(shp_e[i])
        shp_pop1 = np.load(shp_pop[i])
        # print(shp_pop1.shape[0])
        namdtime = shp_pop1.shape[0]
        shp = np.zeros((namdtime,shp_pop1.shape[1]+2))
        shp[:,0] = np.arange(1,namdtime+1)
        shp[:,1] =  shp[:,0]
        shp[:,2:] = shp_pop1
        plot_pop(shp, pop_index, k_labels,  axes1[graph_label[i]], band_labels=['CBM','CBM+1'], text='electron')
    
    #### just plot 300K tdkprop    
    bassel = np.load('part1/vbm/bassel.npy')+1
    kpts = np.load('part1/vbm/kpts.npy')
    # k_index, k_loc, kp_loc = pn.pick_kpts_on_path(kpts, kpath, A, norm=0.001)
    # q_index, q_loc, qp_loc = pn.pick_kpts_on_path(qpts, qpath, A, norm=0.001)
    k_indexs = [[0.333333,0.333333,0],[0.666667,0.666667,0]]
    k_labels = ['+K','-K']
    band_indexs = [1,2]
    pop_index = []
    for b in band_indexs:
        band_data = []
        for k_target in k_indexs:
            k_indices = []
            for i in range(kpts.shape[0]):
                if (periodic_distance(kpts[i], k_target))<0.15 and bassel[i,1] == b:
                    k_indices.append(i)
            band_data.append(k_indices)
        pop_index.append(band_data)
    
    graph_label = ['B','BB']
    shp_e = ['part1/vbm/fssh_e_sh-0.npy','part2/vbm/fssh_e_sh-0.npy']
    shp_pop = ['part1/vbm/fssh_pop_sh-0.npy','part2/vbm/fssh_pop_sh-0.npy']
    ls = ['--','-']
    for i in range(len(shp_pop)):
        shp_e1 = np.load(shp_e[i])
        shp_pop1 = np.load(shp_pop[i])
        namdtime = shp_pop1.shape[0]
        shp = np.zeros((namdtime,shp_pop1.shape[1]+2))
        shp[:,0] = np.arange(1,namdtime+1)
        shp[:,1] = shp[:,0]
        shp[:,2:] = shp_pop1
        plot_pop(shp, pop_index, k_labels,  axes1[graph_label[i]], band_labels=['VBM-1','VBM'],text='hole',order=True)

    plt.savefig('fig4_tot_v4.png', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('fig4_tot_v4.pdf', bbox_inches='tight', pad_inches=0.02)
    # plt.show()

def plot_pop(shp, pop_index, k_labels, ax, band_labels,text,order=False):
    namdtime = shp.shape[0]
    mpl.rcParams.update({
        'axes.unicode_minus': False,
        'axes.labelsize': 12
    })
    all_sum_data = []
    
    # 定义颜色和自旋映射
    color_map = {
        ('+K', 'VBM'): 'red',
        ('-K', 'VBM'): 'blue', 
        ('+K', 'VBM-1'): 'blue',
        ('-K', 'VBM-1'): 'red',
        ('+K', 'CBM'): 'red',
        ('-K', 'CBM'): 'blue',
        ('+K', 'CBM+1'): 'blue',
        ('-K', 'CBM+1'): 'red'
    }
    
    # 使用更短的虚线线段
    line_style_map = {
        'VBM': '-',                    # 实线
        'VBM-1': (0, (1.5, 1.5)),     # 很短的虚线
        'CBM': '-',                    # 实线  
        'CBM+1': (0, (1, 2))          # 点状虚线
    }
    
    # 或者使用点状虚线
    # line_style_map = {
    #     'VBM': '-',           # 实线
    #     'VBM-1': (0, (1, 1)), # 点状虚线
    #     'CBM': '-',           # 实线  
    #     'CBM+1': (0, (1, 2))  # 更稀疏的点
    # }
    
    # 调整线宽 - 虚线可以稍细一些
    linewidth_map = {
        'VBM': 1.5,      
        'VBM-1': 1.3,    # 稍细的虚线
        'CBM': 1.5,      
        'CBM+1': 1.3     # 稍细的虚线
    }
    
    for band_idx, band_data in enumerate(pop_index):
        for k_idx, k_indices in enumerate(band_data):
    # for k_idx, band_data in enumerate(pop_index):
    #     for band_idx, k_indices in enumerate(band_data):
            if not k_indices: 
                continue
            data_columns = [shp[:, i+2] for i in k_indices]
            sum_data = np.sum(data_columns, axis=0)
            all_sum_data.append(sum_data)
            
            band_label = band_labels[band_idx]
            k_label = k_labels[k_idx]
            
            color = color_map.get((k_label, band_label), 'black')
            linestyle = line_style_map.get(band_label, '-')
            linewidth = linewidth_map.get(band_label, 1.5)
            
            if k_label == '+K':
                if color == 'blue':
                    label = r'K$^+_\downarrow$'  
                    label = r'+K$\downarrow$'  
                else:
                    label = r'K$^+_\uparrow$'    
                    label = r'+K$\uparrow$'  
            else:  # -K
                if color == 'blue':
                    label = r'K$^-_\downarrow$'  
                    label = r'-K$\downarrow$' 
                else:
                    label = r'K$^-_\uparrow$'    
                    label = r'-K$\uparrow$'  
            
            ax.plot(shp[:, 0], sum_data, 
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                    solid_capstyle='round',
                    dash_capstyle='round')
            params, params_covariance = curve_fit(linear_func_down,  shp[:, 0][4000:], sum_data[4000:])
            print(f"{k_label} {band_label} decay time: {(0.5-params[1])/params[0]:.2f} fs")
               
    ax.set_xlim(0,8000)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_xticklabels([])
    # ax.set_ylim(0,1.0)
    ax.set_ylabel('Population')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    handles, labels = ax.get_legend_handles_labels()
    if order:
        order1 = [2,3,0,1]
    else:
        order1 = [0,1,2,3]
    ordered_handles = [handles[i] for i in order1]
    ordered_labels = [labels[i] for i in order1]
    handles, labels = ax.get_legend_handles_labels()
    ncol = 3 if len(labels) > 6 else 2  
    if order:
        leg1 = ax.legend(ordered_handles, ordered_labels, 
                    fontsize=9,
                    ncol=ncol,
                    frameon=False,
                    shadow=False,
                    loc='center right',bbox_to_anchor=(0.65, 0.5))
    else:
        leg1 = ax.legend(ordered_handles, ordered_labels, 
            fontsize=9,
            ncol=ncol,
            frameon=False,
            shadow=False,
            loc='upper right',bbox_to_anchor=(1.0, 0.87))
    ax.add_artist(leg1)
    ax.text(0.80, 0.90, text,
        ha="center",
        va="center",
        fontsize=10,
        # family='monospace',
        # fontweight=',
        transform=ax.transAxes)


def plot_ub(axes,vb_file,cb_file):
    # vb_file = sorted(glob('vbm/fssh_pop_sh-0.npy'))
    # cb_file = sorted(glob('cbm/fssh_pop_sh-0.npy'))

    spin_vb = np.load('part1/vbm/vbm_spin150.npy').ravel()
    spin_cb = np.load('part1/cbm/cbm_spin150.npy').ravel()
    orb_vb = np.load('../fig1/vbm/selected_vbm_morb_z.npy').ravel()
    orb_cb = np.load('../fig1/cbm/selected_cbm_morb_z.npy').ravel()
    vb_pop = [np.load(vb_file)]
    cb_pop = [np.load(cb_file)]
    print(len(vb_file))
    # band_labels = ['0','LA','TO2','TA','LO1','LO2']
    #band_labels = ['0','LA','TO2','TA','LO1']
    lss = ['-','--']
    for i in range(len(vb_pop)):
        vb_i = vb_pop[i]
        namdtime = vb_i.shape[0]
        namdtime_array = np.arange(1,namdtime+1)
        vb_i = vb_pop[i]
        cb_i = cb_pop[i]
        # vb_i = vb_i @ spin_vb
        # cb_i = cb_i @ spin_cb
        # a = 3*vb_i + cb_i
        # print(a.shape)
        # moment_spin = vb_i + cb_i
        # moment_valley = 2*vb_i
        moment_spin = vb_i @ spin_vb + cb_i @ spin_cb
        moment_orb = -vb_i @ orb_vb + cb_i @ orb_cb
        # print(a.shape)
        axes.plot(namdtime_array, -moment_spin, 
                linewidth=1.5,
                ls = lss[i],label = r'$M_z^{\mathrm{spin}}$')
        axes.plot(namdtime_array, -moment_orb, 
                linewidth=1.5,
                ls = lss[i],label = r'$M_z^{\mathrm{orb}}$')
        axes.plot(namdtime_array, -moment_spin-moment_orb, 
                linewidth=1.5,
                ls = lss[i],label = r'$M_z$')
    # for i in range(pop_index.shape[0]):
    #     for j in range(pop_index.shape[1]):
    #         index = pop_index[i, j]
    #         ax.plot(shp[:, 0], shp[:, index + 2], label='%s_%s' % (band_labels[i],k_labels[j]))
    axes.set_xlim(0,namdtime)
    new_order = [1, 0, 2]
    handles, labels = axes.get_legend_handles_labels()
    ordered_handles = [handles[i] for i in new_order]
    ordered_labels = [labels[i] for i in new_order]
    axes.legend(handles=ordered_handles, labels=ordered_labels, 
                fontsize=10, frameon=False, shadow=False)
    # ax.set_ylim(0,1.0)    
    axes.set_xlabel('Time (fs)')
    axes.set_ylabel(r"Magnetic moment  ($\mu_B$)")
    axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(2))
    # axes.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    # handles, labels = axes.get_legend_handles_labels()
    # ncol = 3 if len(labels) > 6 else 2  # 根据标签数量自动调整列数
    # axes.legend(handles, labels, 
    #             fontsize=10,
    #             ncol=ncol,
    #             frameon=True,
    #             shadow=True,
    #             loc='best')

def periodic_distance(vec1, vec2):
    delta = np.abs(vec1 - vec2)
    delta = np.minimum(delta, 1 - delta)
    return np.linalg.norm(delta)

def times2index(times, shp):

    tindex = []
    for time in times:
        if (time==0 and shp[0,0]>0): time = shp[0,0]
        index = np.argwhere(shp[:,0]==time)
        if (len(index) > 0):
            tindex.append(index[0][0])
        else:
            print("\nWARNING: No data in SHPROP/PHPROP files " +\
                  "for time=%.0f fs!"%time)

    return np.array(tindex, dtype='int')

# def exp_func_down(x, t):
#     return 1/2 * np.exp(-x/t) + 0.5

# def exp_func_up(x, t):
#     return -1/2 * np.exp(-x/t) + 0.5

# def linear_func_down(x, t):
#     return -1/(2*t)*x + 1.0

# def linear_func_up(x, t):
#     return 1/(2*t)*x

# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + 0.5

# def gaussian_down(x, A, mu, sigma, B):
#     return -A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B

def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B

def linear_func_down(x, k, B):
    return k*x + B

if 'main' in __name__:
    main()
