import sys
import numpy as np
import configparser
import readh5
import readh5shm
from mpi4py import MPI

def ReadEpc(
    comm,PHCUT,EMIN,EMAX,dt,hbar,EFX,EFY,EFZ,LTRANS,LEF,LEPCSHM,LPHSHM
):
    myid = comm.Get_rank()
    nprocs = comm.Get_size()

    conf = configparser.ConfigParser()
    conf.read('config.ini',encoding='utf-8')
    EpcType = conf['epc']['EpcType']
    if EpcType != 'A':
        print('EpcType not supporting!')
        sys.exit()

    inDir = conf['epc']['inDir']+'/'
    bandDir = conf['epc']['bandDir']+'/'
    phononDir = conf['epc']['phononDir']+'/'
    epcDir = conf['epc']['epcDir']+'/'

    atom_str = conf['epc']['atom']
    atom_list = atom_str[1:-1].split(',')
    atom = [int(i) for i in atom_list]
    atomnum = sum(atom)
    nmodes = atomnum*3

    nq_str = conf['epc']['nq']
    nq_list = nq_str[1:-1].split(',')
    nq = np.array([int(i) for i in nq_list],dtype=np.int32)

    IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
    if not IsAllKlist:
        Emin = float(conf['epc']['emin'])
        Emax = float(conf['epc']['emax'])
        bassel = np.load(inDir+bandDir+'/bassel-%d.npy'%nq[0])

    phvalname = conf['epc']['phvalname']
    phname = inDir+phononDir+phvalname
    #phonon = np.load(inDir+phononDir+phvalname)

    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1
    valname = conf['epc']['valname']
    IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
    if IsAllVec:
        IsAllKlist = True
        energy = np.ascontiguousarray(
            np.load(inDir+bandDir+valname)[:,bmin:bmax+1]
        )
    else:
        IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
        if IsAllKlist:
            energy = np.load(inDir+bandDir+valname)
        else:
            bassel = np.load(inDir+bandDir+'/bassel-%d.npy'%nq[0])
            nk = bassel.shape[0]
            ekidx = np.ascontiguousarray(bassel[:,0])
            ebidx = np.ascontiguousarray(bassel[:,1])
            valpname = valname.split('.')[0]+'_p.npy'
            energy = np.load(inDir+bandDir+valpname)

    epcname = inDir+epcDir+'epc_all_p-%d.dat'%nq[0]

    poscar = open(inDir+conf['epc']['poscar_ucell']).readlines()
    abc = np.array([i.split()[0:3] for i in poscar[2:5]],dtype=float)*float(poscar[1].split()[0])

    if IsAllVec or IsAllKlist:
        if LEPCSHM:
            return readh5shm.ReadNpy(
                comm,nmodes,nbands,nq[0],nq[1],nq[2],PHCUT,
                EMIN,EMAX,energy,abc,dt,hbar,EFX,EFY,EFZ,
                LTRANS.encode('utf-8'),phname,
                epcname.encode('utf-8'),LEF,LPHSHM
            )
        else:
            return readh5.ReadNpy(
                comm,nmodes,nbands,nq[0],nq[1],nq[2],PHCUT,
                EMIN,EMAX,energy,abc,dt,hbar,EFX,EFY,EFZ,
                LTRANS.encode('utf-8'),phname,
                epcname.encode('utf-8'),LEF,LPHSHM
            )
    else:
        if LEPCSHM:
            return readh5shm.ReadNpyPart(
                comm,nmodes,nbands,nq[0],nq[1],nq[2],nk,ekidx,ebidx,
                PHCUT,EMIN,EMAX,energy,abc,dt,hbar,EFX,EFY,EFZ,
                LTRANS.encode('utf-8'),phname,
                epcname.encode('utf-8'),LEF,LPHSHM
            )
        else:
            return readh5.ReadNpyPart(
                comm,nmodes,nbands,nq[0],nq[1],nq[2],nk,ekidx,ebidx,
                PHCUT,EMIN,EMAX,energy,abc,dt,hbar,EFX,EFY,EFZ,
                LTRANS.encode('utf-8'),phname,
                epcname.encode('utf-8'),LEF,LPHSHM
            )
