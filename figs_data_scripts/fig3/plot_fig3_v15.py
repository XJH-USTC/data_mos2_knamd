#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import math
from glob import glob
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

mpl.rcParams['axes.unicode_minus'] = False

def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B

def exp_relax(x, A, tau, B):
    # Exponential relaxation over the selected fitting window; tau is in fs.
    return A * np.exp(-(x - x[0]) / tau) + B

def linear_func_down(x, k, B):
    return k*x + B

def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax2 = ax[0, 0] # a: Gamma point
    ax3 = ax[0, 1] # b: Magnitude Bar
    ax1 = ax[1, 0] # c: K point
    ax4 = ax[1, 1] # d: Time Bar
    
    # --- 数据读取逻辑 (完全保留原始逻辑) ---
    vbg_file = sorted(glob('g_8000/vb_*'))
    cbg_file = sorted(glob('g_8000/cb_*'))
    vbk_file = sorted(glob('k/vb_*'))
    cbk_file = sorted(glob('k/cb_*'))
    if vbk_file: vbk_file.pop()
    if cbk_file: cbk_file.pop()
    
    spin_vb = np.load('k/vbm_spin150.npy').ravel()
    spin_cb = np.load('k/cbm_spin150.npy').ravel()
    orb_vb = np.load('../fig1/vbm/selected_vbm_morb_z.npy').ravel()
    orb_cb = np.load('../fig1/cbm/selected_cbm_morb_z.npy').ravel()
    vbg_pop, cbg_pop, vbk_pop, cbk_pop = [], [], [], []
    mag_max, time_up, time_down = [], [], []

    for i in range(len(vbg_file)):
        vbg_pop.append(np.load(vbg_file[i]))
        cbg_pop.append(np.load(cbg_file[i]))
    for i in range(len(vbk_file)):
        vbk_pop.append(np.load(vbk_file[i]))
        cbk_pop.append(np.load(cbk_file[i]))

    # --- 样式与颜色映射 ---
    no_drive_label = 'no phonon\ndriving'
    k_labels = [no_drive_label, 'LA', r'TO$_2$', 'TA', r'LO$_2$']
    g_labels = [no_drive_label, 'LA', r'LO$_2$', r'TO$_2$', r'ZO$_1$']
    color_map = {
        no_drive_label: 'black', 'LA': 'tab:blue', 'TA': 'tab:red',
        r'LO$_2$': 'tab:orange', r'TO$_2$': 'tab:green', r'ZO$_1$': 'tab:purple'
    }

    # --- 绘图: ax1 (K point) ---
    for i in range(len(vbk_pop)):
        vb_i = vbk_pop[i]
        namdtime = vb_i.shape[0]
        namdtime_array = np.arange(1, namdtime+1)
        cb_i = cbk_pop[i]
        moment_spin = vb_i @ spin_vb + cb_i @ spin_cb
        moment_orb = -vb_i @ orb_vb + cb_i @ orb_cb
        # vb_i = vb_i @ spin_vb
        # cb_i = cb_i @ spin_cb
        # a_val = -(3*vb_i + cb_i)
        a_val = -(moment_orb + moment_spin)
        mag_max.append(np.max(abs(a_val)))
        
        current_label = k_labels[i]
        ax1.plot(namdtime_array, a_val, lw=1.5, label=current_label, color=color_map[current_label])
        
        # 拟合数据填充 (保持原逻辑)
        if i == 1 or i == 2: 
            params, _ = curve_fit(gaussian, namdtime_array[:1000], a_val[:1000])
            time_up.append(abs(params[2]))
        elif i == 3:
            params, _ = curve_fit(gaussian, namdtime_array[:1500], a_val[:1500])
            time_up.append(abs(params[2]))
        elif i == 0:
            params, _ = curve_fit(gaussian, namdtime_array[:1900], a_val[:1900])
            time_up.append(abs(params[2]))
        else:
            params, _ = curve_fit(gaussian, namdtime_array[:2000], a_val[:2000])
            time_up.append(abs(params[2]))
        
        if i == 0: params, _ = curve_fit(linear_func_down, namdtime_array[3000:], a_val[3000:])
        elif i == 1: params, _ = curve_fit(linear_func_down, namdtime_array[3000:], a_val[3000:])
        elif i == 2: params, _ = curve_fit(linear_func_down, namdtime_array[3000:], a_val[3000:])
        elif i == 3: params, _ = curve_fit(linear_func_down, namdtime_array[3000:], a_val[3000:])
        else: params, _ = curve_fit(linear_func_down, namdtime_array[3000:], a_val[3000:])
        time_down.append(-1*params[1]/params[0])

    # --- 绘图: ax2 (Gamma point) ---
    for i in range(len(vbg_pop)):
        vb_i = vbg_pop[i]
        namdtime = vb_i.shape[0]
        namdtime_array = np.arange(1, namdtime+1)
        cb_i = cbg_pop[i]
        # vb_i = vb_i @ spin_vb
        # cb_i = cb_i @ spin_cb
        # a_val = -(3*vb_i + cb_i)
        moment_spin = vb_i @ spin_vb + cb_i @ spin_cb
        moment_orb = -vb_i @ orb_vb + cb_i @ orb_cb
        a_val = -(moment_orb + moment_spin)
        mag_max.append(np.max(abs(a_val)))
        
        current_label = g_labels[i]
        ax2.plot(namdtime_array, a_val, lw=1.5, label=current_label, color=color_map[current_label])

        if i != 0:
            if i == 1:
                fit_n = 1000
            elif i == 2 or i == 3: 
                fit_n = 1000
            elif i == 4: 
                fit_n = 1000

            fit_x = namdtime_array[:fit_n]
            fit_y = a_val[:fit_n]
            if i == 1:
                p0 = (fit_y[0] - fit_y[-1], 300.0, fit_y[-1])
                params, _ = curve_fit(exp_relax, fit_x, fit_y, p0=p0, maxfev=10000)
                time_up.append(abs(params[1]))
            else:
                y0 = fit_y[0]
                x0 = fit_x[0]
                gaussian_from_start = lambda x, sigma, B: B + (y0 - B) * np.exp(-(x - x0)**2 / (2 * sigma**2))
                params, _ = curve_fit(gaussian_from_start, fit_x, fit_y, p0=(300.0, fit_y[-1]), maxfev=10000)
                time_up.append(abs(params[0]))
            if i == 1: 
                params, _ = curve_fit(linear_func_down, namdtime_array[4000:], a_val[4000:])
            else: 
                params, _ = curve_fit(linear_func_down, namdtime_array[5000:], a_val[5000:])
            time_down.append(-1*params[1]/params[0])

    # --- 绘图: ax3 (Magnitude - 图b) ---
    del mag_max[5] # 移除重复的 0
    categories = [
        no_drive_label, 'LA@K', r'TO$_2$@K', 'TA@K', r'LO$_2$@K',
        r'LA@$\Gamma$', r'LO$_2$@$\Gamma$', r'TO$_2$@$\Gamma$', r'ZO$_1$@$\Gamma$'
    ]
    c_colors = [color_map[no_drive_label], color_map['LA'], color_map[r'TO$_2$'], color_map['TA'], color_map[r'LO$_2$'],
                color_map['LA'], color_map[r'LO$_2$'], color_map[r'TO$_2$'], color_map[r'ZO$_1$']]
    
    bars1 = ax3.bar(categories, mag_max, color=c_colors, edgecolor='black', linewidth=0.5)
    
    # 标注 K 和 Gamma 区域
    ax3.axvline(x=0.5, color='black', linestyle='--', lw=1, alpha=0.6)
    ax3.axvline(x=4.5, color='black', linestyle='--', lw=1, alpha=0.6)
    ax3.text(2.5, 4.35, 'K-point phonons', ha='center', fontweight='bold', fontsize=14)
    ax3.text(6.5, 4.35, r"$\mathbf{\Gamma}$-point phonons", ha='center', fontweight='bold', fontsize=14)

    # 动态红框标注整个 LA(Gamma) 柱子
    la_idx = 5
    target_bar = bars1[la_idx]
    y_min = 1.8 # 坐标轴底端
    highlight_rect = Rectangle(
        (target_bar.get_x(), y_min), 
        target_bar.get_width(), 
        target_bar.get_height() - y_min, 
        linewidth=2, edgecolor='red', facecolor='none', zorder=10
    )
    ax3.add_patch(highlight_rect)
    # 动态红框标注整个 TO2(K) 柱子
    to2_idx = 2
    target_bar = bars1[to2_idx]
    y_min = 1.8 # 坐标轴底端
    highlight_rect = Rectangle(
        (target_bar.get_x(), y_min), 
        target_bar.get_width(), 
        target_bar.get_height() - y_min, 
        linewidth=2, edgecolor='red', facecolor='none', zorder=10
    )
    ax3.add_patch(highlight_rect)
    # --- 绘图: ax4 (Time - 图d) ---
    x = np.arange(len(categories))
    bar_width = 0.38
    bars_u = ax4.bar(x - bar_width/2, time_up, width=bar_width, color=c_colors, label=r'$\tau_1$ (enhancement)')
    bars_d = ax4.bar(x + bar_width/2, time_down, width=bar_width, facecolor='none', edgecolor=c_colors, linewidth=1.2, hatch='///', label=r'$\tau_2$ (relaxation)')

    # 仿照图b，对 d 图中的 K-point 与 Gamma-point phonons 做分区
    ax4.axvline(x=0.5, color='black', linestyle='--', lw=1, alpha=0.6)
    ax4.axvline(x=4.5, color='black', linestyle='--', lw=1, alpha=0.6)

    # 时间标注，非0显示
    for bs in [bars_u, bars_d]:
        for b in bs:
            h = b.get_height()
            if h > 0:
                ax4.text(b.get_x() + b.get_width()/2, h + 5, f'{h:1.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    #################################################################################
    bar_u_la = bars_u[la_idx]
    bar_d_la = bars_d[la_idx]
    # 计算红框边界：从 tau1 柱子的左边缘到 tau2 柱子的右边缘
    x_start = bar_u_la.get_x()
    total_width = (bar_d_la.get_x() + bar_d_la.get_width()) - x_start
    max_h = max(bar_u_la.get_height(), bar_d_la.get_height())
    
    rect_d = Rectangle((x_start, 0), total_width, max_h + 10, # 加10fs余量避免顶到文字
                       linewidth=2, edgecolor='red', facecolor='none', zorder=10)
    ax4.add_patch(rect_d)
    #################################################################################
    bar_u_la = bars_u[to2_idx]
    bar_d_la = bars_d[to2_idx]
    # 计算红框边界：从 tau1 柱子的左边缘到 tau2 柱子的右边缘
    x_start = bar_u_la.get_x()
    total_width = (bar_d_la.get_x() + bar_d_la.get_width()) - x_start
    max_h = max(bar_u_la.get_height(), bar_d_la.get_height())
    
    rect_d = Rectangle((x_start, 0), total_width, max_h + 10, # 加10fs余量避免顶到文字
                       linewidth=2, edgecolor='red', facecolor='none', zorder=10)
    ax4.add_patch(rect_d)
    #################################################################################
    time_label_y = max(max(time_up), max(time_down)) + 1000
    ax4.text(2.5, time_label_y, 'K-point phonons', ha='center', fontweight='bold', fontsize=14)
    ax4.text(6.5, time_label_y, r"$\mathbf{\Gamma}$-point phonons", ha='center', fontweight='bold', fontsize=14)

    # --- 统一设置标签与细节 ---
    # a图 (ax2)
    ax2.set_ylabel(r"$\mathit{M}_z$ ($\mu_B$)", fontsize=16)
    ax2.set_xlabel('Time [fs]', fontsize=14)
    ax2.set_ylim(-4.4, -1.5)
    ax2.set_xlim(0, 4000)
    ax2.text(0.5, 0.95, r"$\mathbf{\Gamma}$-point phonons", transform=ax2.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    
    # b图 (ax3)
    ax3.set_ylabel(r"|$\mathit{M}_z$| ($\mu_B$)", fontsize=16)
    ax3.set_xlabel('Phonon mode', fontsize=14)
    ax3.set_ylim(2.8, 4.5)
    for b in bars1:
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.2f}', ha='center', fontweight='bold', fontsize=8)

    # c图 (ax1)
    ax1.set_ylabel(r"$\mathit{M}_z$ ($\mu_B$)", fontsize=16)
    ax1.set_xlabel('Time [fs]', fontsize=14)
    ax1.set_xlim(0, 8000)
    # ax1.set_ylim(-2.5, -0.4)
    ax1.text(0.5, 0.95, r"$\mathbf{K}$-point phonons", transform=ax1.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    # d图 (ax4)
    ax4.set_ylabel('Time (fs)', fontsize=16)
    ax4.set_xlabel('Phonon mode', fontsize=14)
    ax4.set_ylim(0, time_label_y + 2000)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=18)
    ax4.legend(loc='upper right', frameon=False,bbox_to_anchor=(1.0, 0.9))
    x_pos = np.arange(len(categories))

    # a, b, c, d 序列标注
    for i, sub in enumerate([ax2, ax3, ax1, ax4]):
        sub.text(0.02, 0.93, "acbd"[i], transform=sub.transAxes, fontsize=18, fontweight='bold', family='monospace')
        sub.grid(True, linestyle='--', alpha=0.3, which='both')
        if sub is ax1:
            handles, labels = sub.get_legend_handles_labels()
            legend_order = [1, 2, 0, 3, 4]
            sub.legend(
                [handles[j] for j in legend_order],
                [labels[j] for j in legend_order],
                loc='lower right',
                fontsize=10,
                ncol=2,
                frameon=False
            )
        if sub is ax2:
            handles, labels = sub.get_legend_handles_labels()
            legend_order = [1, 2, 0, 3, 4]
            sub.legend(
                [handles[j] for j in legend_order],
                [labels[j] for j in legend_order],
                loc='upper right',
                fontsize=10,
                ncol=2,
                frameon=False,
                bbox_to_anchor=(1.0, 0.85)
            )
        if i == 1 or i == 3: # ax3 和 ax4
            sub.set_xticks(x_pos)
            sub.set_xticklabels(categories, rotation=18)
            # 修改特定索引刻度的颜色
            labels = sub.get_xticklabels()
            labels[la_idx].set_color('red') 
            labels[la_idx].set_fontweight('bold')
            labels[to2_idx].set_color('red') 
            labels[to2_idx].set_fontweight('bold')

    fig.tight_layout(pad=1.2)
    fig.savefig('fig3_v15.png', dpi=600)


if __name__ == '__main__':
    main()
