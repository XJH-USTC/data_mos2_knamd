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
from matplotlib.ticker import AutoMinorLocator


mpl.rcParams['axes.unicode_minus'] = False
################################################################################
################################################################################

def main():
    fig = plt.figure(
    figsize=(9.6, 5.4),
    dpi=480,
    )
    ################################################################################
    # fig1    
    axes1 = fig.subplot_mosaic(
        [[0]],
        gridspec_kw={
            "bottom": 0.45,
            "top": 0.95,
            "left": 0.07,
            "right": 0.32,
            "wspace": 0.1
            # "hspace": 0.2,
        },
        empty_sentinel=-1,
    )
    axes1[0].text(0.03, 0.97, "{}".format('a'),
                ha="center",
                va="center",
                fontsize=14,
                family='monospace',
                fontweight='bold',
                # rotation=45,
                transform=axes1[0].transAxes,
                # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
        )
    #### add toc graph
    img_path = './spin_valley_part1.3.png' 
    img = mpimg.imread(img_path)
    axes6 = fig.subplot_mosaic(
        [[0]],
        gridspec_kw={
            "bottom": 0.0,
            "top": 0.40,
            "left": 0.07,
            "right": 0.32,
            "wspace": 0.05
            # "hspace": 0.2,
        },
        empty_sentinel=-1,
    )
    ax_toc = axes6[0]
    ax_toc.imshow(img)
    ax_toc.axis('off')
    ax_toc.set_aspect('equal')
    ax_toc.text(0.03, 0.97, "{}".format('b'),
                ha="center",
                va="center",
                fontsize=14,
                family='monospace',
                fontweight='bold',
                # rotation=45,
                transform=ax_toc.transAxes,
                # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
        )
    ## add hexgonal brillouin zone    
    axes5 = fig.subplot_mosaic(
        [[0]],
        gridspec_kw={
            "bottom": 0.60,
            "top": 0.80,
            "left": 0.07,
            "right": 0.32,
            "wspace": 0.1
        },
        # height_ratios=[1, 2.0],
        # width_ratios=[1, 1],
    )
    # axes5[0].text(0.18, 0.90, "{}".format('b'),
    #             ha="center",
    #             va="center",
    #             fontsize=14,
    #             family='monospace',
    #             fontweight='bold',
    #             # rotation=45,
    #             transform=axes5[0].transAxes,
    #             # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
    #     )
    plot_hexagonal(axes5)
    
    bandsfile = 'band.dat'
    spnfile = 'band.dat.3'
    bandsoutfile = 'bands.out'
    kpts, eig = readEig(bandsfile)
    spn = readspn(spnfile)
    # spn *= -1 #
    kaxis = kptsCoordinate(kpts)
    hskpts = HSkpts(bandsoutfile)

    plotBands2(kaxis, eig, spn, hskpts, ax=axes1[0], Efermi=-5.0893, emin=-2, emax=2)
    
    ################################################################################
    # fig2


    axes2 = fig.subplot_mosaic(
        [['A'],['B'],['C']],
        gridspec_kw={
            "bottom": 0.09,
            "top": 0.95,
            "left": 0.40,
            "right": 0.62,
            "wspace": 0.1,
            "hspace": 0.2,
            "height_ratios": [1, 1, 1.1]
        },
        # height_ratios=[1, 2.0],
        # width_ratios=[1, 1],
    )
    for i,j in enumerate(axes2):
        label = 'cde'[i]
        axes2[j].text(0.06, 0.91, "{}".format(label),
                    ha="center",
                    va="center",
                    fontsize=14,
                    family='monospace',
                    fontweight='bold',
                    # rotation=45,
                    transform=axes2[j].transAxes,
                    # bbox=dict(boxstyle='round, pad=0.2', facecolor='wheat', alpha=0.5)
            )
    n_cols, n_rows = 1, 3
    layout = [[f'R{r}C{c}' for c in range(n_cols)] for r in range(n_rows)]
    axes3 = fig.subplot_mosaic(
        layout,
        gridspec_kw={
            "bottom": 0.09,
            "top": 0.98,
            "left": 0.66,
            "right": 0.81,
            "wspace": 0.1,
            "hspace": -0.1,
        },
        # height_ratios=[1, 2.0],
        # width_ratios=[1, 1],
    )
    for i, key in enumerate(axes3):
        if i == 0 :
            label = "{}".format(chr(ord('f') + i))
            axes3[key].text(0.06, 0.92, label,
                            ha="center",
                            va="center",
                            fontsize=14,
                            family='monospace',
                            fontweight='bold',
                            transform=axes3[key].transAxes)

    axes4 = fig.subplot_mosaic(
        layout,
        gridspec_kw={
            "bottom": 0.09,
            "top": 0.98,
            "left": 0.84,
            "right": 0.99,
            "wspace": -0.2,
            "hspace": -0.1,
        },
        # height_ratios=[1, 2.0],
        # width_ratios=[1, 1],
    )
    for i, key in enumerate(axes4):
        if i ==0:
            label = "{}".format(chr(ord('g') + i))
            axes4[key].text(0.06, 0.92, label,
                            ha="center",
                            va="center",
                            fontsize=14,
                            family='monospace',
                            fontweight='bold',
                            transform=axes4[key].transAxes)

    ## plot ub
    plot_ub(axes2['C'])
    ##

    kpath = np.array([ # for TDBAND.png
        [0.50000, 0.50000, 0.00000],
        [0.66667, 0.66667, 0.00000],
        [1.00000, 0.00000, 0.00000],
        [0.33333, 0.33333, 0.00000],
        [0.50000, 0.50000, 0.00000]
        ])
    
    kplabels = 'mkgkm'
    qpath = kpath # for TDPH.png
    qplabels = kplabels
    A = np.load('cbm/A.npy')
    a1, a2, a3 = (A[0], A[1], A[2])
    bassel = np.load('cbm/bassel.npy')+1
    en = np.load('cbm/en.npy')
    kpts = np.load('cbm/kpts.npy')
    Enk = np.load('cbm/Enk.npy')
    qpts = np.load('cbm/qpts.npy')
    phen = np.load('cbm/phen.npy')
    # k_index, k_loc, kp_loc = pn.pick_kpts_on_path(kpts, kpath, A, norm=0.001)
    # q_index, q_loc, qp_loc = pn.pick_kpts_on_path(qpts, qpath, A, norm=0.001)
    spin_cb = np.load('cbm/cbm_spin150.npy')
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
    
    shp_e = [np.load('cbm/shp_e.npy')]
    shp_pop = [np.load('cbm/shp_pop.npy')]
    for i in range(len(shp_pop)):
        shp_e1 = shp_e[i]
        shp_pop1 = shp_pop[i]
        # print(shp_pop1.shape[0])
        namdtime = shp_pop1.shape[0]
        shp = np.zeros((namdtime,shp_pop1.shape[1]+2))
        shp[:,0] = np.arange(1,namdtime+1)
        shp[:,1] =  shp[:,0]
        shp[:,2:] = shp_pop1
        plot_pop(shp, pop_index, k_labels, axes2['A'], band_labels=['CBM','CBM+1'], text='electron')
    
    #### just plot 300K tdkprop    
    times = [0,400,1500]
    plot_tdkprop(kpts, shp, times, axes3, n_cols, n_rows, fig, title='electron', axis='xy', cmap = 'hot_r')


    A = np.load('vbm/A.npy')
    a1, a2, a3 = (A[0], A[1], A[2])
    bassel = np.load('vbm/bassel.npy')+1
    en = np.load('vbm/en.npy')
    kpts = np.load('vbm/kpts.npy')
    Enk = np.load('vbm/Enk.npy')
    qpts = np.load('vbm/qpts.npy')
    phen = np.load('vbm/phen.npy')
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
    
    shp_e = [np.load('vbm/shp_e.npy')]
    shp_pop = [np.load('vbm/shp_pop.npy')]
    ls = ['--','-']
    for i in range(len(shp_pop)):
        shp_e1 = shp_e[i]
        shp_pop1 = shp_pop[i]
        namdtime = shp_pop1.shape[0]
        shp = np.zeros((namdtime,shp_pop1.shape[1]+2))
        shp[:,0] = np.arange(1,namdtime+1)
        shp[:,1] = shp[:,0]
        shp[:,2:] = shp_pop1
        plot_pop(shp, pop_index, k_labels, axes2['B'], band_labels=['VBM-1','VBM'], text='hole', order=True)

    #### just plot 300K tdkprop  
    times = [0,400,1500]
    plot_tdkprop(kpts, shp, times, axes4, n_cols, n_rows, fig, title='hole', axis='xy', cmap='PuBu')
    plt.savefig('fig1_v12.png')
    plt.savefig('fig1_v12.pdf')
    # plt.show()


def HSkpts(filename):
    '''
    Get x coordinate of high-symmetry k points from bands.out file.
    '''
    with open(filename) as f:
        lines = f.readlines()

    hskpts = []
    for line in lines:
        if 'high-symmetry point' in line:
            hskpts.append(float(line.split()[-1]))
    return hskpts
    

def readEig(filename):
    '''
    Read eigen value and k points from bands data file.
    '''
    with open(filename) as f:
        lines = f.readlines()

    headline = lines[0].replace(',','')
    nks  = int(headline.split()[-2])
    nbnd = int(headline.split()[2])
    kpts = np.zeros((nks, 3), dtype=float)
    eig  = np.zeros((nks, nbnd), dtype=float)

    # n is number of energy lines for each k point.
    if nbnd%10==0:
        n = nbnd // 10
    else:
        n= nbnd // 10 + 1

    numline = 1
    for i in range(nks):
        lineK = lines[numline]; numline += 1
        kpts[i] = np.array(lineK.split())
        states = []
        for j in range(n):
            lineE = lines[numline]; numline += 1
            states.extend(lineE.split())
        eig[i] = np.array(states)
#    print(kpts.shape, eig.shape)
#    print(np.min(eig[:,24]))
#    print(np.min(eig[:,23]))
#    input()
    return kpts, eig

def readspn(filename):
    '''
    Read spin value from bands data file.
    '''
    with open(filename) as f:
        lines = f.readlines()

    headline = lines[0].replace(',','')
    nks  = int(headline.split()[-2])
    nbnd = int(headline.split()[2])
    kpts = np.zeros((nks, 3), dtype=float)
    spin  = np.zeros((nks, nbnd), dtype=float)

    # n is number of energy lines for each k point.
    if nbnd%10==0:
        n = nbnd // 10
    else:
        n= nbnd // 10 + 1

    numline = 1
    for i in range(nks):
        lineK = lines[numline]; numline += 1
        kpts[i] = np.array(lineK.split())
        states = []
        for j in range(n):
            lineE = lines[numline]; numline += 1
            states.extend(lineE.split())
        spin[i] = np.array(states)
    return spin

def kptsCoordinate(kpts):
    '''
    Calculate k axis coordinate of k points.
    '''
    kaxis = [0.0]
    for i in range(1,len(kpts)):
        s = kpts[i] - kpts[i-1]
        length = (s[0]**2 + s[1]**2 + s[2]**2)**0.5
        kaxis.append( kaxis[-1] + length )

    return kaxis


def HSkptLabels(kpath):
    '''
    Instruct k axis tick labels.
    '''
    kticklabels = []
    for s in kpath:
        if s=='g' or s=='G':
            s = r'$\Gamma$'
        else:
            s = s.upper()
        kticklabels.append(s)
    return kticklabels

def plotBands(kaxis, eig, hskpts, ax, Efermi=-5.0893, emin=-2, emax=2):
    '''
    Plot bands and save as Band.png fig.
    '''
    # fig, ax = plt.subplots()
    # fig.set_size_inches(3.6,4.8)
    # mpl.rcParams['axes.unicode_minus'] = False

    eig -= Efermi
    kaxis = np.array(kaxis)
    for i in range(eig.shape[1]):
        ax.plot(kaxis, eig[:,i], 'black', lw=1)
    # ax.plot([kaxis[0],kaxis[-1]], [0,0], 'k', ls='-', lw=0.5)
    for kpt in hskpts:
        ax.plot([kpt,kpt], [emin,emax], 'k', ls='-', lw=0.5,alpha=0.5)
    ax.set_xlim(kaxis[0], kaxis[-1])
    ax.set_ylim(emin, emax)
    ax.set_xticks(hskpts)
    # ax.set_xticklabels(HSkptLabels(kpath))
    ax.set_xticklabels([r'M',r'+K',r'$\Gamma$',r'-K',r'M'])
    ax.set_ylabel('Energy (eV)')

def plotBands2(kaxis, eig, spin, hskpts, ax, Efermi=-5.0893, emin=-2,emax=2):
    '''
    Plots band structure with spin-colored line segments using LineCollection.
    '''

    # Shift eigenvalues by Fermi energy
    eig_shifted = eig - Efermi
    nbands = eig_shifted.shape[1]

    # Dynamic y-axis limits
    # eig_min, eig_max = np.min(eig_shifted), np.max(eig_shifted)
    # padding = (eig_max - eig_min) * padding_frac
    # emin, emax = eig_min - padding, eig_max + padding

    # Convert kaxis to numpy array
    kaxis = np.array(kaxis)

    # Create colormap for spin values
    cmap = plt.get_cmap('bwr')
    norm = plt.Normalize(vmin=np.min(spin)*2, vmax=np.max(spin)*2)

    # Plot each band with LineCollection
    for i in range(nbands):
        # 生成线段数据
        points = np.array([kaxis, eig_shifted[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 计算线段颜色（使用相邻两点spin值的平均）
        spin_avg = (spin[1:, i] + spin[:-1, i]) / 2.0 * 2
        lc = LineCollection(segments, colors=cmap(norm(spin_avg)), linewidth=1.5)
        
        # 先画灰色背景线
        ax.plot(kaxis, eig_shifted[:, i], color='lightgray', lw=0.8, zorder=1)
        # 再覆盖彩色线条
        ax.add_collection(lc)

    # Fermi level line
    # ax.axhline(0, color='k', linestyle='--', lw=0.5, zorder=3)

    # High-symmetry k-point lines
    for kpt in hskpts:
        ax.axvline(kpt, color='k', ls='--', lw=0.5, zorder=2)

    # 设置坐标轴范围
    ax.set_xlim(kaxis[0], kaxis[-1])
    ax.set_ylim(emin, emax)

    # 添加colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    ax_cbar = divider.append_axes('top',size='3%', pad=0.02)
    cbar = plt.colorbar(sm, cax=ax_cbar,
                    ticks=[-1.0,-0.5,0.0,0.5, 1.0],
                    orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.set_xticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax.set_xticks(hskpts)
    ax.set_xticklabels([r'M',r'-K',r'$\Gamma$',r'+K',r'M'])
    ax.set_ylim(emin, emax)
    ax.set_ylabel('Energy (eV)')
    # ax.set_xticklabels([label.strip('$') for label in kpath])

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
    
    ax.set_xlim(0,8000)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_xticklabels([])
    ax.set_ylim(-0.1,1.2)
    ax.set_ylabel('Population')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    handles, labels = ax.get_legend_handles_labels()
    if order:
        order = [2,3,0,1]
    else:
        order = [0,1,2,3]
    ordered_handles = [handles[i] for i in order]
    ordered_labels = [labels[i] for i in order]
    ncol = 3 if len(labels) > 6 else 2  
    leg1 = ax.legend(ordered_handles, ordered_labels, 
                fontsize=9,
                ncol=ncol,
                frameon=False,
                shadow=False,
                loc='center right',bbox_to_anchor=(1.0, 0.53))
    ax.add_artist(leg1)
    ax.text(0.80, 0.90, text,
            ha="center",
            va="center",
            fontsize=10,
            # family='monospace',
            # fontweight=',
            transform=ax.transAxes)

def preprocess_population(kpts, pop, target_kpoint=np.array([1/3, 1/3, 0]), k_radius=0.001):
    ntimes, nkpts = pop.shape
    processed_pop = np.copy(pop)
    

    unique_kpts, unique_indices = np.unique(kpts, axis=0, return_index=True)
    
    for it in range(2):
        current_pop = pop[it, :]
        T = np.array([[1, 0], [0.5, np.sqrt(3)/2]])
        distances = np.linalg.norm(kpts[:, :2]@T - target_kpoint[:2]@T, axis=1)
        target_mask = distances <= k_radius
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) > 0:
            max_pop_value = np.max(current_pop)
            max_k_idx = np.argmax(current_pop)
            # target_pops = current_pop[target_indices]
            # max_in_target_idx = np.argmax(target_pops)
            # max_k_idx = target_indices[max_in_target_idx]
            # max_k_point = kpts[max_k_idx]
            # max_pop_value = current_pop[max_k_idx]
            
            
            avg_pop = max_pop_value / len(target_indices)
            
            processed_pop[it, max_k_idx] = 0
            processed_pop[it, target_indices] += avg_pop
            
    
    return processed_pop

def plot_tdkprop(kpts, shp, times, axes, ncol, nrow, fig, title, axis='xy', cmap='hot_r'):
    axdict = {'x':0, 'y':1, 'z':2}
    kax = [axdict[s] for s in axis]
    tindex = times2index(times, shp)
    times = shp[tindex,0]
    nts = len(times)
    pop = shp[tindex,2:]
    print(kpts.shape)
    print(pop.shape)
    pop = preprocess_population(kpts, pop, target_kpoint=np.array([1/3, 1/3, 0]), k_radius=0.01)
    # 计算颜色范围
    cmin = np.min(pop[pop > 0.0])
    cmin = 2e-4
    cmax = 1.0
    cmin = max(cmin, 1e-10)  
    norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
    # cmap = 'hot_r'
    
    # 定义晶胞基矢（六角格子）
    L = 1.0
    cell = L * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2.]])
    
    # 计算六边形顶点
    cc = L * np.sqrt(3) / 3 * np.exp(1j * np.pi * (1./6 + 1./3 * np.arange(7)))
    hex_vertices = np.c_[cc.real, cc.imag]
    
    # 计算扩展区域（正方形边界）
    hex_xmin, hex_xmax = hex_vertices[:,0].min(), hex_vertices[:,0].max()
    hex_ymin, hex_ymax = hex_vertices[:,1].min(), hex_vertices[:,1].max()
    square_size = max(hex_xmax-hex_xmin, hex_ymax-hex_ymin) * 1.25
    square_center_x = (hex_xmin + hex_xmax) / 2
    square_center_y = (hex_ymin + hex_ymax) / 2
    square_xmin = square_center_x - square_size/2
    square_xmax = square_center_x + square_size/2
    square_ymin = square_center_y - square_size/2
    square_ymax = square_center_y + square_size/2
    
    # 设置主标题
    axes[f'R{0}C{0}'].set_title(title, pad=8)
    
    # 创建ScalarMappable用于颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for it in range(nts):
        ax = axes[f'R{it//ncol}C{it%ncol}']
        
        # 添加时间标签
        ax.text(0.95, 0.95, f'{times[it]:.0f} fs',
                ha="right", va="top", fontsize='small',
                fontweight='bold', transform=ax.transAxes)
        
        # 添加正方形边框
        square = plt.Rectangle(
            (square_xmin, square_ymin), 
            square_size, 
            square_size,
            linewidth=1.0, edgecolor='black', 
            facecolor='none', zorder=9
        )
        ax.add_patch(square)
        
        # 添加六边形边框
        hex_border = Polygon(hex_vertices, lw=1.0, edgecolor='k',
                             facecolor='none', zorder=10)
        ax.add_patch(hex_border)
        
        # 按占据数排序（大的点在顶部）
        sort_idx = np.argsort(pop[it, :])
        
        all_points = []
        all_values = []
        
        # 扩展范围确保覆盖整个正方形区域
        # 计算需要多少重复晶胞覆盖正方形
        cell_range_x = int(np.ceil(square_size / np.linalg.norm(cell[0])))
        cell_range_y = int(np.ceil(square_size / np.linalg.norm(cell[1])))
        
        # 循环扩展足够的晶胞
        for ii in range(-cell_range_x, cell_range_x+1):
            for jj in range(-cell_range_y, cell_range_y+1):
                # 计算平移后的分数坐标
                kx_trans = kpts[:, kax[0]] + ii
                ky_trans = kpts[:, kax[1]] + jj
                
                # 转换为直角坐标
                kx_cart = kx_trans * cell[0, 0] + ky_trans * cell[1, 0]
                ky_cart = kx_trans * cell[0, 1] + ky_trans * cell[1, 1]
                
                # 收集所有点和值
                all_points.append(np.column_stack([kx_cart, ky_cart]))
                all_values.append(pop[it, :])
        
        # 合并所有数据
        all_points = np.vstack(all_points)
        all_values = np.concatenate(all_values)
        
        # 仅绘制在正方形区域内的点
        in_square_mask = (
            (all_points[:, 0] >= square_xmin) & 
            (all_points[:, 0] <= square_xmax) &
            (all_points[:, 1] >= square_ymin) & 
            (all_points[:, 1] <= square_ymax)
        )
        points_in_square = all_points[in_square_mask]
        values_in_square = all_values[in_square_mask]
        
        # 绘制所有点（不应用裁剪）
        sorted_idx = np.argsort(values_in_square)
        sc = ax.scatter(points_in_square[sorted_idx, 0], points_in_square[sorted_idx, 1],
                        c=values_in_square[sorted_idx], s=10, lw=0,
                        cmap=cmap, norm=norm)
        
        # 设置坐标范围为正方形边界
        ax.set_xlim(square_xmin, square_xmax)
        ax.set_ylim(square_ymin, square_ymax)
        ax.set_aspect('equal')
        ax.axis('off')

    h_cb = 0.015
    l, b, w, h = ax.get_position().bounds
    cb_ax = fig.add_axes([l, b-0.5*h_cb-0.02, w, h_cb])
    cbar = plt.colorbar(sc, cax=cb_ax,orientation='horizontal')
    tick_positions = [10**-3, 10**-2, 10**-1, 10**0]
    tick_labels = [
        # r'$10^{-4}$',
        r'$10^{-3}$',
        r'$10^{-2}$',
        r'$10^{-1}$',
        r'$10^{0}$'
    ]
    # 设置刻度
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    # 可选：设置刻度标签的字体大小
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Carrier density', fontsize=10, labelpad=5)
def plot_ub(axes):
    vb_file = sorted(glob('vbm/shp_pop.npy'))
    cb_file = sorted(glob('cbm/shp_pop.npy'))

    spin_vb = np.load('vbm/vbm_spin150.npy').ravel()
    spin_cb = np.load('cbm/cbm_spin150.npy').ravel()
    orb_vb = np.load('vbm/selected_vbm_morb_z.npy').ravel()
    orb_cb = np.load('cbm/selected_cbm_morb_z.npy').ravel()
    vb_pop = []
    cb_pop = []

    for i in range(len(vb_file)):
        vb_pop.append(np.load(vb_file[i]))
        cb_pop.append(np.load(cb_file[i]))
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
    axes.set_ylim(-4.2,0.6)
    new_order = [1, 0, 2]
    handles, labels = axes.get_legend_handles_labels()
    ordered_handles = [handles[i] for i in new_order]
    ordered_labels = [labels[i] for i in new_order]
    axes.legend(ordered_handles, ordered_labels, 
            ncol=3, fontsize=9, frameon=False, shadow=False,
            loc='upper right', bbox_to_anchor=(1.02,1.02),
            columnspacing=0.8, handlelength=1.2)
    # ax.set_ylim(0,1.0)    
    axes.set_xlabel('Time (fs)')
    axes.set_ylabel(r" Magnetic moment  ($\mu_B$)")
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

def exp_func_down(x, t):
    return 1/2 * np.exp(-x/t) + 0.5

def exp_func_up(x, t):
    return -1/2 * np.exp(-x/t) + 0.5

def linear_func_down(x, t):
    return -1/(2*t)*x + 1.0

def linear_func_up(x, t):
    return 1/(2*t)*x

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + 0.5

def gaussian_down(x, A, mu, sigma, B):
    return -A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B

def plot_hexagonal(axes):
    L = 1.0
    
    # 计算六边形顶点
    cc = L * np.sqrt(3) / 3 * np.exp(1j * np.pi * (1./6 + 1./3 * np.arange(7)))
    hex_vertices = np.c_[cc.real, cc.imag]
    
    # 定义高对称点Γ、M、K的位置 (六角晶格中的标准位置)
    Gamma = np.array([0.0, 0.0])  # Γ点(中心点)
    
    # M点 - 六边形边的中点（两个M点中选一个）
    M = (hex_vertices[1] + hex_vertices[2]) / 2
    
    # K点 - 六边形顶点
    K_plus = hex_vertices[2]  # +K点
    K_minus = hex_vertices[1]  # -K点

    
    # 计算视图范围(以六边形为中心)
    margin = 0.1 * L
    xmin, xmax = hex_vertices[:,0].min()-margin, hex_vertices[:,0].max()+margin
    ymin, ymax = hex_vertices[:,1].min()-margin, hex_vertices[:,1].max()+margin
        
    ax = axes[0]
    
    # 添加六边形边框
    hex_border = Polygon(hex_vertices, lw=1.5, edgecolor='k',
                            facecolor='none', zorder=10)
    ax.add_patch(hex_border)
    
    # 绘制高对称点标记
    # Γ点
    ax.scatter([Gamma[0]], [Gamma[1]], s=30, color='black', marker='o', zorder=12)
    ax.text(Gamma[0], Gamma[1]+0.05, r'$\mathbf{\Gamma}$', 
            fontsize=8, ha='center', va='bottom', color='black', 
            weight='bold', zorder=12)
    
    # M点
    ax.scatter([M[0]], [M[1]], s=30, color='black', marker='o', zorder=12)
    ax.text(M[0], M[1]+0.05, r'$\mathbf{M}$', 
            fontsize=8, ha='center', va='bottom', color='black', 
            weight='bold', zorder=12)
    
    # +K点
    ax.scatter([K_plus[0]], [K_plus[1]], s=30, color='black', marker='o', zorder=12)
    ax.text(K_plus[0], K_plus[1]+0.05, r'$\mathbf{+K}$', 
            fontsize=8, ha='center', va='bottom', color='black', 
            weight='bold', zorder=12)
    
    # -K点
    ax.scatter([K_minus[0]], [K_minus[1]], s=30, color='black', marker='o', zorder=12)
    ax.text(K_minus[0], K_minus[1]+0.20, r'$\mathbf{-K}$', 
            fontsize=8, ha='center', va='top', color='black', 
            weight='bold', zorder=12)
    
    # 添加Γ到各点的连线
    # ax.plot([Gamma[0], M[0]], [Gamma[1], M[1]], 
    #         'gray', linestyle='--', lw=0.8, alpha=0.5)
    # ax.plot([Gamma[0], K_plus[0]], [Gamma[1], K_plus[1]], 
    #         'gray', linestyle='--', lw=0.8, alpha=0.5)
    # ax.plot([Gamma[0], K_minus[0]], [Gamma[1], K_minus[1]], 
    #         'gray', linestyle='--', lw=0.8, alpha=0.5)
    
    # 添加轴标签（可选）
    # ax.set_xlabel('kx', fontsize=12)
    # ax.set_ylabel('ky', fontsize=12)
    
    # 设置坐标范围
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')

if 'main' in __name__:
    main()
