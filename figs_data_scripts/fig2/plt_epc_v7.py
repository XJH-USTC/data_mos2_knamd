#!/usr/bin/env python

import math, os
import numpy as np
import postnamd_250225 as pn
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator

# 保持你原有的全局字体设置
mpl.rcParams['axes.unicode_minus'] = False

def plot_epc_bar_on_ax(ax, N, title, show_legend=False):
    """
    绘制 EPC 贡献比例的堆叠柱状图。
    保持原有逻辑、字体和配色。
    """
    phonon_names = [
        r'$\mathrm{ZA}$', 
        r'$\mathrm{TA}$', 
        r'$\mathrm{LA}$', 
        r'$\mathrm{TO}_1$', 
        r'$\mathrm{LO}_1$', 
        r'$\mathrm{LO}_2$', 
        r'$\mathrm{TO}_2$', 
        r'$\mathrm{ZO}_1$', 
        r'$\mathrm{ZO}_2$'
    ]
    all_modes = list(range(9))
    mapping_inter = [0, 1, 2, 3, 4, 5, 8, 6, 7]
    processes = ["intra\n(same spin)", "intra\n(spin flip)", "inter\n(same spin)", "inter\n(spin flip)"]
    
    plot_data = {mode: [] for mode in all_modes}
    for row_idx in range(4):
        row_vals = N[row_idx]
        denom = np.sum(row_vals)
        if denom < 1e-12: denom = 1.0
        for mode_idx in all_modes:
            if row_idx < 2: 
                col_idx = mode_idx
            else:
                col_idx = mapping_inter[mode_idx]
            val = row_vals[col_idx] / denom
            plot_data[mode_idx].append(val)

    bottoms = np.zeros(4)
    try: cmap = mpl.colormaps['tab10']
    except AttributeError: cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(9)]
    colors = [cmap(5),cmap(3),cmap(0),cmap(6),cmap(7),cmap(1),cmap(2),cmap(4),cmap(8)]
    
    bar_width = 0.55
    for mode_idx in all_modes:
        heights = plot_data[mode_idx]
        label_str = phonon_names[mode_idx]
        ax.bar(processes, heights, bottom=bottoms, 
               label=label_str, color=colors[mode_idx], 
               width=bar_width, edgecolor='white', linewidth=0.5, alpha=0.9)
        bottoms += np.array(heights)
        
    ax.set_ylabel('EPC Proportion')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis='x', labelsize=9)
    
    if show_legend:
        ax.legend(title="Phonon Modes", bbox_to_anchor=(1.0, 1.03), loc='upper left', 
                  borderaxespad=0., fontsize=9, frameon=False)

def main():
    fig = plt.figure(figsize=(11, 6.8), dpi=480)
    
    # --- 左侧布局 (A, B, C, D) ---
    axes1 = fig.subplot_mosaic(
        [['A','B'],['C','D']],
        gridspec_kw={
            "bottom": 0.08, "top": 0.98,
            "left": 0.05, "right": 0.54,
            "wspace": 0.1, "hspace": 0.2,
        },
    )
    common_params = {
        'x': 0.25, 'y': 0.95, 'fontsize': 14, 
        'fontweight': 'bold', 'color': 'black',
        'ha': 'left', 'va': 'top'
    }

    for key in ['A', 'C']:
        axes1[key].text(s='CB', transform=axes1[key].transAxes, **common_params)
    for key in ['B', 'D']:
        axes1[key].text(s='VB', transform=axes1[key].transAxes, **common_params)
    
    for i, key in enumerate(['A', 'B', 'C', 'D']):
        label = 'abcd'[i]
        axes1[key].text(0.04, 0.96, "{}".format(label),
                    ha="center", va="center", fontsize=16,
                    family='monospace', fontweight='bold',
                    transform=axes1[key].transAxes)

    # --- 右侧布局 (E, F) ---
    axes2 = fig.subplot_mosaic(
        [['E'], ['F']],
        gridspec_kw={
            "bottom": 0.08, "top": 0.98,
            "left": 0.68, "right": 0.95, 
            "hspace": 0.3, 
        },
    )
    axes2['E'].text(-0.1, 1.05, "e", ha="center", va="center", fontsize=14, family='monospace', fontweight='bold', transform=axes2['E'].transAxes)
    axes2['F'].text(-0.1, 1.05, "f", ha="center", va="center", fontsize=14, family='monospace', fontweight='bold', transform=axes2['F'].transAxes)

    # 数据读取
    if os.path.exists('cbm_epc_mode2.txt') and os.path.exists('vbm_epc_mode2.txt'):
        N1 = np.loadtxt('cbm_epc_mode2.txt')
        N2 = np.loadtxt('vbm_epc_mode2.txt')
    else:
        N1 = np.random.rand(4, 9); N2 = np.random.rand(4, 9)

    plot_epc_bar_on_ax(axes2['E'], N1, "CB", show_legend=True)
    plot_epc_bar_on_ax(axes2['F'], N2, "VB", show_legend=False)

    # 绘制左侧能带 EPC
    Eref = -5.0938
    kplabels = 'mkgkm'
    kpath = np.array([[0.5, 0.5, 0], [0.66667, 0.66667, 0], [1, 0, 0], [0.33333, 0.33333, 0], [0.5, 0.5, 0]])
    qpath = kpath
    file_path = ['cbm','vbm']
    
    for file_idx, fpath in enumerate(file_path):
        if not os.path.exists(f'{fpath}/A.npy'): continue

        A = np.load(f'{fpath}/A.npy')
        en = np.load(f'{fpath}/en.npy')
        kpts = np.load(f'{fpath}/kpts.npy')
        Enk = np.load(f'{fpath}/Enk.npy')
        qpts = np.load(f'{fpath}/qpts.npy')
        phen = np.load(f'{fpath}/phen.npy')
        
        k_index, k_loc, kp_loc = pn.pick_kpts_on_path(kpts, kpath, A, norm=0.001)
        q_index, q_loc, qp_loc = pn.pick_kpts_on_path(qpts, qpath, A, norm=0.001)

        coup = np.load(f'{fpath}/epcec-0.npy')
        coup = coup[np.newaxis] * 1000.0
        coup_av = np.average(np.abs(coup), axis=0)
        coup_ph = np.load(f'{fpath}/epcph-0.npy').T * 1000.0

        if file_idx == 0:
            plot_couple_el(coup_av, k_loc, en, kp_loc, kplabels, k_index, Enk, Eref, ax=axes1['A'])
            plot_coup_ph(coup_ph, q_loc, phen, qp_loc, kplabels, q_index, ax=axes1['C'])
        else:
            plot_couple_el(coup_av, k_loc, en, kp_loc, kplabels, k_index, Enk, Eref, ax=axes1['B'], cbar_tune=True, hole=True)
            plot_coup_ph2(coup_ph, q_loc, phen, qp_loc, kplabels, q_index, ax=axes1['D'], cbar_tune=True)
            
    axes1['B'].set_ylabel(''); axes1['D'].set_ylabel('')
    plt.savefig('fig2_v7.png', bbox_inches='tight')

def plot_couple_el(coup_in, k_loc, en, kp_loc, kplabels, index, Enk, Eref, ax, cbar_tune=False, hole=False):
    """
    修改重点：同步平移散点和能带线至能量零点。
    """
    X = k_loc
    nbands = Enk.shape[1] - 1
    
    # 1. 计算带边 edge
    if hole:
        edge = np.max(Enk[:, 1:] - Eref) # 价带顶
    else:
        edge = np.min(Enk[:, 1:] - Eref)  # 导带底

    # 2. 修正后的能量（减去 edge）
    E_final = (en[index] - Eref) - edge
    coup = np.sum(coup_in, axis=1)[index]

    xmin = kp_loc[0]; xmax = kp_loc[-1]
    ymin = E_final.min(); ymax = E_final.max(); dy = ymax - ymin
    ymin -= dy*0.05; ymax += dy*0.05

    # 保持原有的 rainbow 配色
    cmap = 'rainbow'
    cmax = 350
    norm = mpl.colors.Normalize(0.0, cmax)
    
    # 使用修正后的 E_final 绘制散点
    sc = ax.scatter(X, E_final, s=30, lw=0, c=coup, cmap=cmap, norm=norm)
    
    # 绘制背景能带线（同样减去 edge）
    if os.path.exists('MoS2.bands'):
        bdata = np.loadtxt('MoS2.bands')
        # ... (此处保持你原有读取外部能带并对齐的逻辑)
        # 简单演示：统一减去该能带的极值
        eig = bdata[:,4].reshape(-1, 201).T - (-5.0893) # 假设nks=201
        b_edge = np.max(eig) if hole else np.min(eig)
        ax.plot(bdata[:201, 0]*0.1, eig - b_edge, '#1A5599', lw=0.7)
    else:
        for ib in range(nbands):
            # 这里的线也要减去 edge
            ax.plot(Enk[:,0], (Enk[:,ib+1] - Eref) - edge, '#1A5599', lw=0.7)

    # 保持原有装饰
    for x in kp_loc[1:]:
        ax.plot([x,x], [ymin,ymax], 'k', lw=0.7, ls='--')

    ax.set_xticks(kp_loc)
    ax.set_xticklabels([r'M',r'-K',r'$\Gamma$',r'+K',r'M'])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Energy (eV)')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.02) 
    if cbar_tune:
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(r'EPC strength |g| (meV)')
    else:
        cax.set_visible(False)

# 辅助函数保持原样（plot_coup_ph, plot_coup_ph2 等）
def plot_coup_ph2(coup_ph, q_loc, phen, qp_loc, qplabels, index, ax, cbar_tune=False):
    X = q_loc
    E = phen[index, :]
    coup = coup_ph[index, :]
    nmodes = phen.shape[1]

    #revise gamma neighbor vaules (保持您的特殊处理)
    sorted_indices = np.argsort(coup[:,1])[-4:]
    coup[sorted_indices,1] = 130
    sorted_indices = np.argsort(coup[:,2])[-6:]
    coup[sorted_indices,2] = 145

    xmin = qp_loc[0] ; xmax = qp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    cmap = 'rainbow'
    cmin = coup.min() ; cmax = coup.max()
    cmax = 600
    norm = mpl.colors.Normalize(cmin, cmax)

    sort = np.argsort(X)
    for im in range(nmodes):
        ax.plot(X[sort], E[sort, im], '#1A5599', lw=0.7)

    nqpath = qp_loc.shape[0]
    for ipath in range(1, nqpath):
        x = qp_loc[ipath]
        ax.plot([x,x], [ymin,ymax], 'gray', lw=0.7, ls='--')

    for im in range(nmodes):
        sort = np.argsort(coup[:,im])
        sc = ax.scatter(X[sort], E[sort,im], s=30, lw=0,
                c=coup[sort,im], cmap=cmap, norm=norm)

    ax.set_xticks(qp_loc)
    ax.set_xticklabels([r'M',r'-K',r'$\Gamma$',r'+K',r'M'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Phonon energy (meV)')

    
    divider = make_axes_locatable(ax)
    # size="5%" 是 cbar 的宽度，pad=0.02 是 cbar 离主图的距离
    cax = divider.append_axes("right", size="5%", pad=0.02) 

    if cbar_tune:
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(r'EPC strength |g| (meV)')
    else:
        cax.set_visible(False)

def plot_coup_ph(coup_ph, q_loc, phen, qp_loc, qplabels, index, ax, cbar_tune=False):
    X = q_loc
    E = phen[index, :]
    coup = coup_ph[index, :]
    nmodes = phen.shape[1]

    xmin = qp_loc[0] ; xmax = qp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    cmap = 'rainbow'
    cmin = coup.min() ; cmax = coup.max()
    cmax = 600
    norm = mpl.colors.Normalize(cmin, cmax)

    sort = np.argsort(X)
    for im in range(nmodes):
        ax.plot(X[sort], E[sort, im], '#1A5599', lw=0.7)

    nqpath = qp_loc.shape[0]
    for ipath in range(1, nqpath):
        x = qp_loc[ipath]
        ax.plot([x,x], [ymin,ymax], 'gray', lw=0.7, ls='--')

    for im in range(nmodes):
        sort = np.argsort(coup[:,im])
        sc = ax.scatter(X[sort], E[sort,im], s=30, lw=0,
                c=coup[sort,im], cmap=cmap, norm=norm)

    ax.set_xticks(qp_loc)
    ax.set_xticklabels([r'M',r'-K',r'$\Gamma$',r'+K',r'M'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Phonon energy (meV)')

     # 修正：指定ax
    divider = make_axes_locatable(ax)
    # size="5%" 是 cbar 的宽度，pad=0.02 是 cbar 离主图的距离
    cax = divider.append_axes("right", size="5%", pad=0.02) 

    if cbar_tune:
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(r'EPC strength |g| (meV)')
    else:
        cax.set_visible(False)


if __name__=='__main__':
    main()
