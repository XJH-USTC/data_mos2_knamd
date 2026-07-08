#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import Args
import readh5
import readh5shm
import readepc
import sys
import numpy as np
import mpi4py
from mpi4py import MPI
import os
import CalFunc
import CalFuncShm

def SurfHop():
    # INICON read
    inicon = np.loadtxt('INICON',dtype=np.int32)-1
    if len(inicon.shape) == 1:
        inicon = inicon.reshape(1,-1)
    if inicon.shape[0] < Args.NSAMPLE:
        print('Sample number in INICON is samller than NSAMPLE!')
        sys.exit()

    nel = (inicon.shape[1]-1)//2
    step_s = inicon[:,0]

    # get world comm and split comm to shm_comm
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    nprocs_shm = shm_comm.Get_size()
    shm_num = nprocs//nprocs_shm

    # get split num = GCD(shm_num,NSAMPLE)
    split_num = 1
    tmp1 = min(shm_num,Args.NSAMPLE)
    tmp2 = max(shm_num,Args.NSAMPLE)
    while (1):
        tmp2 = tmp2 - tmp1
        if tmp2 == 0:
            split_num = tmp1
            break
        else:
            tmp3 = min(tmp1,tmp2)
            tmp4 = max(tmp1,tmp2)
            tmp1 = tmp3
            tmp2 = tmp4
    iprocs = nprocs//split_num

    # split comm
    color= myid//iprocs
    isample = Args.NSAMPLE//split_num
    comm_split = comm.Split(color=color,key=myid)
    # comm_split rank_id and nprocs
    myid_split = comm_split.Get_rank()

    for i in range(split_num):
        if color == i:
            if not Args.LHDF5:
                nk,nk_a,nq,n_p,nmodes,nbands,ekidx,ebidx,kqidx_p,\
                k1kidxE,k_proc,k_proc_num,energy,phonon,epc_a\
                = readepc.ReadEpc(
                    comm_split,Args.PHCUT/1000.0,Args.EMIN,Args.EMAX,
                    Args.dt,Args.hbar,Args.EFX,Args.EFY,Args.EFZ,
                    Args.LTRANS,Args.LEF,Args.LEPCSHM,Args.LPHSHM
                )
            else:
                if not Args.LEPCSHM:
                    nk,nk_a,nq,n_p,nmodes,nbands,ekidx,ebidx,kqidx_p,\
                    k1kidxE,k_proc,k_proc_num,energy,phonon,epc_a\
                    = readh5.ReadH5(
                        comm_split,
                        (Args.EPMDIR+'/'+Args.EPMPREF).encode('utf-8'),
                        Args.NPARTS,Args.PHCUT,Args.EMIN,Args.EMAX,
                        Args.nqx,Args.nqy,Args.nqz,Args.dt,
                        Args.hbar,Args.EFX,Args.EFY,Args.EFZ,
                        Args.LTRANS.encode('utf-8'),Args.LEF,Args.LPHSHM
                    )
                else:
                    nk,nk_a,nq,n_p,nmodes,nbands,ekidx,ebidx,kqidx_p,\
                    k1kidxE,k_proc,k_proc_num,energy,phonon,epc_a\
                    = readh5shm.ReadH5(
                        comm_split,
                        (Args.EPMDIR+'/'+Args.EPMPREF).encode('utf-8'),
                        Args.NPARTS,Args.PHCUT,Args.EMIN,Args.EMAX,
                        Args.nqx,Args.nqy,Args.nqz,Args.dt,
                        Args.hbar,Args.EFX,Args.EFY,Args.EFZ,
                        Args.LTRANS.encode('utf-8'),Args.LEF,Args.LPHSHM
                    )
            #np.save('epc-%d-%d.npy'%(myid,k_proc[myid]),epc_a)

            if color == 0 and myid_split == 0:
                #np.save('k1kidxE.npy',k1kidxE)
                #np.save('epc_a.npy',epc_a)
                Args.WriteInp(nbands,nk,n_p)
                np.save(Args.namddir+'/bassel.npy',\
                        np.vstack([ekidx[0:nk_a],ebidx[0:nk_a]]).T)
                np.save(Args.namddir+'/energy.npy',energy)
            nk_proc = k_proc_num[myid_split]

            for j in range(i*isample,(i+1)*isample):
                starttime = MPI.Wtime()
                istep_s = step_s[j]
                ikstate_s = inicon[j,np.arange(1,nel*2+1,2)]
                ibstate_s = inicon[j,np.arange(2,nel*2+2,2)]
                istate_s = np.zeros((nel),dtype=np.int32)
                for iel in range(nel):
                    istate_s[iel] = CalFunc.GetIniIdx(\
                        ekidx,ebidx,ikstate_s[iel],ibstate_s[iel],nk_a
                    )
                    if istate_s[iel] < 0 and myid_split == 0:
                        print('Initial energy out of [EMIN,EMAX]!')
                        sys.exit()
                if not Args.LEPCSHM:
                    CalFunc.fssh(
                        comm_split,Args.namddir,j,istep_s,istate_s,
                        Args.NTRAJ,Args.NSW,Args.NELM,Args.KbT,Args.edt,
                        Args.hbar,Args.SIGMA,Args.dt,k_proc,k_proc_num,
                        nk_a,nq,n_p,nmodes,nel,kqidx_p,k1kidxE,epc_a,
                        energy,phonon,Args.LHOLE,Args.LSPLIT,Args.LPHSHM,
                        Args.BANDDEG,Args.EFTYPE.encode('utf-8'),
                        Args.EFSTART,Args.EFTIME
                    )
                else:
                    CalFuncShm.fssh(
                        comm_split,Args.namddir,j,istep_s,istate_s,
                        Args.NTRAJ,Args.NSW,Args.NELM,Args.KbT,Args.edt,
                        Args.hbar,Args.SIGMA,Args.dt,k_proc,k_proc_num,
                        nk_a,nq,n_p,nmodes,nel,kqidx_p,k1kidxE,epc_a,
                        energy,phonon,Args.LHOLE,Args.LSPLIT,Args.LPHSHM,
                        Args.BANDDEG,Args.EFTYPE.encode('utf-8'),
                        Args.EFSTART,Args.EFTIME
                    )

                endtime = MPI.Wtime()
                if myid_split == 0:
                    print("FSSH time in sample %d: %.6fs"\
                          %(j,endtime-starttime))

