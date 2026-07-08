#!/usr/bin/env python

import numpy as np
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parameters Setting
emin = -2.5
emax =  5
#prefix = 'In2Se3_ZB'
Efermi = -5.0893
figname = 'Band.png'
kpath = 'mkgkm'

def main():
    
    bandsfile = 'band.dat'
    spnfile = 'band.dat.3'
#    prefix + '.bands'
    kpts, eig = readEig(bandsfile)
    print(kpts.shape, eig.shape)
    spn = readspn(spnfile)
    print(spn.shape)

    kaxis = kptsCoordinate(kpts)
    bandsoutfile = 'bands.out'
    hskpts = HSkpts(bandsoutfile)

    plotBands(kaxis, eig, spn, hskpts, Efermi)


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

def plotBands(kaxis, eig, spin, hskpts, Efermi):
    '''
    Plot bands and save as Band.png fig.
    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(3.6,4.8)
    mpl.rcParams['axes.unicode_minus'] = False

    eig -= Efermi
    print(len(kaxis),eig.shape,spin.shape)
    kaxis  = np.array(kaxis)
    for i in range(eig.shape[1]):
        ax.scatter(kaxis, eig[:, i], c=spin[:, i], cmap='bwr', )

    ax.plot([kaxis[0],kaxis[-1]], [0,0], 'k', ls='--', lw=0.5)
    for kpt in hskpts:
        ax.plot([kpt,kpt], [emin,emax], 'k', ls='--', lw=0.5)
    ax.set_xlim(kaxis[0], kaxis[-1])
    ax.set_ylim(emin, emax)
    ax.set_xticks(hskpts)
    ax.set_xticklabels(HSkptLabels(kpath))

    plt.tight_layout()
    plt.savefig(figname, dpi=360)

def plotBands(kaxis, eig, spin, hskpts, Efermi, figname='Band.png', padding_frac=0.05):
    '''
    Plots band structure with spin-colored line segments using LineCollection.
    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(3.6, 4.8)
    mpl.rcParams['axes.unicode_minus'] = False

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
    ax.axhline(0, color='k', linestyle='--', lw=0.5, zorder=3)

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
    # 设置ticks
    ax.set_xticks(hskpts)
    ax.set_xticklabels(HSkptLabels(kpath))
    ax.set_xticklabels(['M','-K',r'$\Gamma$','K','M'])
    ax.set_ylim(emin, emax)
    # ax.set_xticklabels([label.strip('$') for label in kpath])
    
    plt.tight_layout()
    plt.savefig(figname, dpi=360)
    plt.close()

if __name__=='__main__':
    main()
