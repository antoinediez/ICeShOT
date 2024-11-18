"""
Cell sorting in 3D
============================================

This script simulates 3D cell aggregates with the following weighted :math:`L^2` cost and force for the cell :math:`i`

.. math::

    c(x,x_i) = \\frac{\gamma_{i0}}{R_i} |x - x_i|^2

.. math:: 

    F_{i\\leftarrow j} = \\int_{\\Gamma_{ij}} \\left(\\gamma_{ij} |\kappa| + \\frac{\eta_{ij}}{|x_i - x_j|} \\right) \\vec{n}\\mathrm{d}\\sigma

.. math::

    F_i = \\sum_{j=0}^N F_{i\\leftarrow j} - \\frac{1}{N_i}\\sum_{k\\in\\mathscr{N}_i} F_{k\\leftarrow 0}
    
where :math:`\\gamma_{ij}` and :math:`\\eta_{ij}` are surface tension parameters, :math:`R_i` is the radius of cell :math:`i`, :math:`\\Gamma_{ij}` is the interface between cells :math:`i` and :math:`j`, :math:`\kappa` is the local mean curvature and :math:`\\vec{n}` is the inward normal of cell :math:`\\mathscr{L}_i`.


In this script, there are two cell types (indexed by :math:`b` and :math:`o`) and the surface tension parameters only depend on the type. Varying them lead to various cell sorting phenomena which can be classified according to the following ratios

.. math::

    \\overline{\\eta} = \\frac{\\eta_{ob}}{\\eta_{oo}},\\quad \\overline{\\gamma} = \\frac{\\gamma_o}{\\gamma_b},\\quad \\overline{k} = \\frac{\\gamma_b\\eta_{oo}}{\\gamma_o\\eta_{bb}}


Representative situations are obtained as follow: 

**Separation** : :math:`\\overline{\\eta} = 3, \\quad \\overline{\\gamma} = 2,\\quad \\overline{k} = 1`

.. video:: ../../_static/SMV11_3Dsorting_separation.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400

**Checkerboard** : :math:`\\overline{\\eta} = 0.3, \\quad \\overline{\\gamma} = 2,\\quad \\overline{k} = 1`


.. video:: ../../_static/SMV12_3Dsorting_checkerboard.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400


**Internalization** : :math:`\\overline{\\eta} = 3, \\quad \\overline{\\gamma} = 2,\\quad \\overline{k}\\overline{\\gamma}\\overline{\\eta} = 1`


.. video:: ../../_static/SMV13_3Dsorting_internalization.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400

**Engulfment with initial segregation** : :math:` \\overline{\\eta} = 3, \\quad \\overline{\\gamma} = 2,\\quad \\overline{k}\\overline{\\gamma}\\overline{\\eta} = 1`

.. video:: ../../_static/SMV14_3Dsorting_engulfment.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400


Note: this script only save the mesh data, that can then be loaded in VTK, PyVista, Paraview etc. 
"""

# sphinx_gallery_thumbnail_path = '_static/3Dengulfment.png'

import os
import sys
sys.path.append("..")
import time
import pickle
import math
import torch
import numpy as np
from matplotlib import colors
from matplotlib.colors import ListedColormap
from iceshot import cells
from iceshot import costs
from iceshot import OT
from iceshot.OT import OT_solver
from iceshot import plot_cells
from iceshot import sample
from iceshot import utils
import tifffile as tif
import pyvista as pv
import vtk as vtk
from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output, _update_alg
from typing import Literal, Optional, cast
from pyvista.core.utilities.arrays import FieldAssociation, set_default_active_scalars


pv.start_xvfb()
pv.set_jupyter_backend('static')

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
  
def run_simu(params,title=None,init="uniform"):

    ot_algo = OT.LBFGSB

    N = 128
    N1 = 64
    
    # N = 96
    # N1 = 24

    M = 256
    dim = 3

    R0 = 1.0
    R00 = 0.055
    RN = torch.ones(N)*R00
    vol_x = 4./3.*math.pi*RN**3
    
    if init == "uniform":
        seeds = 0.5 + 0.14*2.*(torch.rand((N,dim))-0.5)
    elif init == "split":
        seeds1 = 0.4 + 0.1*2.*(torch.rand((N1,dim))-0.5)
        seeds2 = 0.6 + 0.1*2.*(torch.rand((N-N1,dim))-0.5)
        seeds = torch.cat((seeds1,seeds2),dim=0)

    #================ SURFACE TENSION PARAMETERS ==================#

    gb = params["gb"]
    g12 = params["g12"]
    g11 = params["g11"]
    g22 = params["g22"]
    g10 = params["g10"]
    g20 = params["g20"]
    b12 = params["b12"]
    b11 = params["b11"]
    b22 = params["b22"]
    

    print("===============================================================")
    print("Surface Tension Parameters",flush=True)
    print(f"g12={g12}",flush=True)
    print(f"g11={g11}",flush=True)
    print(f"g22={g22}",flush=True)
    print(f"g10={g10}",flush=True)
    print(f"g20={g20}",flush=True)
    print(f"b12={b12}",flush=True)
    print(f"b11={b11}",flush=True)
    print(f"b22={b22}",flush=True)

    print("Compaction Tension Parameters",flush=True)
    print(f"k10={0.5*b11/g10}",flush=True)
    print(f"k20={0.5*b22/g20}",flush=True)
    print(f"k12={0.5*b11/g12}",flush=True)
    
    
    r_b1 = params["r_b1"]
    r_g1 = params["r_g1"]
    r_k1 = params["r_k1"]
    print("Ratios",flush=True)
    print(f"r_b1 = {r_b1}")
    print(f"r_g1 = {r_g1}")
    print(f"r_k1 = {r_k1}")

    if title is None:
        simu_name = f"simu_3Dsorting_b_{r_b1}_g_{r_g1}_k_{r_k1}"
    else:
        simu_name = f"simu_3Dsorting_" + title
    os.mkdir(simu_name)
    os.mkdir(simu_name+"/frames")
    os.mkdir(simu_name+"/data")
    print("===============================================================")


    #===============================================#

    tau = 0.0
    sc = R0/RN
    sc[N1:] *= g20/g10

    source = sample.sample_grid(M,dim=dim)

    simu = cells.Cells(
        seeds=seeds,source=source,
        vol_x=vol_x,extra_space="void",jct_method='Kmin'
    )


    cost_params = {
        "p" : 2,
        "scaling" : "constant",
        "C" : 1.0
    }

    solver = OT_solver(
        n_sinkhorn=800,n_sinkhorn_last=2000,n_lloyds=10,s0=2.0,
        cost_function=costs.power_cost,cost_params=cost_params
    )

    T = 4.0
    dt = 0.0003   # This is too small! 
    plot_every = 20
    save_every = 1
    t = 0.0
    t_iter = 0
    t_plot = 0


    #===========================================================#

    def radius(simu):
        return torch.sqrt(simu.volumes[:-1]/math.pi) if simu.d==2 else (simu.volumes[:-1]/(4./3.*math.pi)) ** (1./3.)


    def compute_mesh(img):
        img = np.pad(img,1,mode='constant',constant_values=-2.0)
        vol = pv.wrap(img)
        alg = vtk.vtkSurfaceNets3D()
        set_default_active_scalars(vol)  # type: ignore
        field, scalars = vol.active_scalars_info  # type: ignore

        # args: (idx, port, connection, field, name)
        alg.SetInputArrayToProcess(0, 0, 0, field.value, scalars)  
        alg.SetInputData(vol)
        alg.GenerateValues(simu.N_cells, 0, simu.N_cells-1)

        # Suppress improperly used INFO for debugging messages in vtkSurfaceNets3D
        verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
        _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_OFF)
        _update_alg(alg, False, 'Performing Labeled Surface Extraction')
        # Restore the original vtkLogger verbosity level
        _vtk.vtkLogger.SetStderrVerbosity(verbosity)
        surfaces = cast(pv.PolyData, pv.wrap(alg.GetOutput()))
        surfaces = surfaces.smooth_taubin(n_iter=100, pass_band=0.05, normalize_coordinates=True)
        surfaces = surfaces.compute_normals(consistent_normals=True,
                                            auto_orient_normals=True,
                                            flip_normals=True,
                                            non_manifold_traversal=False)
        surfaces = surfaces.compute_cell_sizes()
        surfaces["Curvature"] = surfaces.curvature()
        surfaces["Particle"] = surfaces["BoundaryLabels"].min(axis=1)*(surfaces["BoundaryLabels"].min(axis=1)>=0) + surfaces["BoundaryLabels"].max(axis=1)*(surfaces["BoundaryLabels"].min(axis=1)<0)
        surfaces.set_active_scalars("Particle")
        surfaces = surfaces.point_data_to_cell_data()
        return surfaces

    def extract_stuff(surfaces,M=M,simu=simu,eps=None):
        normals = torch.tensor(surfaces["Normals"])
        normals /= torch.norm(normals,dim=1)[:,None]
        lab = torch.tensor(surfaces["BoundaryLabels"])
        area = torch.tensor(surfaces["Area"])/((M+2)**2)
        curv = torch.tensor(surfaces["Curvature"])*(M+2)
        centers = torch.tensor((surfaces.cell_centers().points - 1.0/(M+2))/M)
        return good_stuff(simu,(normals, lab, area, curv, centers),eps=eps)

    def good_stuff(simu,stuff,eps=None):
        return reorient_normals(simu,stuff,eps=eps)

    def belongs_to(simu,x):
        M = round(simu.M_grid ** (1/simu.d))
        ijk = torch.floor(x*M).type(torch.long)
        ijk = torch.clamp(ijk,0,M-1)
        lab = ijk[:,0]*M**2 + ijk[:,1]*M + ijk[:,2]
        labels = simu.labels[lab]
        labels[labels > simu.N_cells-1] = -1.0
        return labels

    def reorient_normals(simu,stuff,eps=None):
        # normals should go from lab[:,0] to lab[:,1]
        normals, lab, area, curv, centers = stuff
        if eps is None:
            eps = 3.0/((simu.M_grid)**(1./simu.d))
        test_m = centers - eps*normals
        test_p = centers + eps*normals
        lab_test_m = belongs_to(simu,test_m)
        lab_test_p = belongs_to(simu,test_p)
        tm_fst = lab_test_m == lab[:,0]
        tm_scd = lab_test_m == lab[:,1]
        tp_fst = lab_test_p == lab[:,0]
        tp_scd = lab_test_p == lab[:,1]
        tm_out = ((test_m.max(dim=1).values>1) | (test_m.min(dim=1).values<0))
        tp_out = ((test_p.max(dim=1).values>1) | (test_p.min(dim=1).values<0))    
        out = (tm_out & tp_scd) | (tm_out & tp_fst) | (tm_fst & tp_out) | (tm_scd & tp_out)
        
        good = (((tm_fst) & (tp_scd)) | ((tm_scd) & (tp_fst)) | out)
        
        to_reorient = ((tm_scd) & (tp_fst)) | (tp_out)
        normals[to_reorient,:] *= -1
        curv[to_reorient] *= -1
        return normals[good], lab[good], area[good], curv[good], centers[good]
        

    def compute_forces(simu,normals,lab,area,curv,centers):
        N = len(simu.x)
        F = torch.zeros_like(simu.x)
        g_ij = torch.zeros(len(lab))
        g_ij[(lab[:,0]<N1) & (lab[:,1]>=N1)] = g12
        g_ij[(lab[:,1]<N1) & (lab[:,0]>=N1)] = g12
        g_ij[(lab[:,0]<N1) & (lab[:,1]<N1)] = g11
        g_ij[(lab[:,0]>=N1) & (lab[:,1]>=N1)] = g22

        g_ij[(lab[:,0]==-1) & (lab[:,1]<N1)] = g10
        g_ij[(lab[:,0]==-1) & (lab[:,1]>=N1)] = g20
        g_ij[(lab[:,1]==-1) & (lab[:,0]<N1)] = g10
        g_ij[(lab[:,1]==-1) & (lab[:,0]>=N1)] = g20

        g_ij[(lab[:,0]==-2) | (lab[:,1]==-2)] = gb
        
        
        b_ij = torch.zeros(len(lab))
        b_ij[(lab[:,0]<N1) & (lab[:,1]>=N1)] = b12
        b_ij[(lab[:,1]<N1) & (lab[:,0]>=N1)] = b12
        b_ij[(lab[:,0]<N1) & (lab[:,1]<N1)] = b11
        b_ij[(lab[:,0]>=N1) & (lab[:,1]>=N1)] = b22

        b_ij[(lab[:,0]==-1) & (lab[:,1]<N1)] = 0
        b_ij[(lab[:,0]==-1) & (lab[:,1]>=N1)] = 0
        b_ij[(lab[:,1]==-1) & (lab[:,0]<N1)] = 0
        b_ij[(lab[:,1]==-1) & (lab[:,0]>=N1)] = 0

        b_ij[(lab[:,0]==-2) | (lab[:,1]==-2)] = gb
        
        
        for i in range(N):
            fst = lab[:,0] == i
            scd = lab[:,1] == i
            
            # Curvature force
            
            F_crv_fst = (-normals[fst,:]*curv[fst,None].abs()*area[fst,None]*g_ij[fst,None]).sum(0) 
            F_crv_scd = (normals[scd,:]*curv[scd,None].abs()*area[scd,None]*g_ij[scd,None]).sum(0)
            F_crv = F_crv_fst + F_crv_scd
            
            # Boundary force
            fst_bnd = fst & (lab[:,1]==-2)
            scd_bnd = scd & (lab[:,0]==-2)
            F_bnd_fst = (-normals[fst_bnd,:]*area[fst_bnd,None]*g_ij[fst_bnd,None]).sum(0)
            F_bnd_scd = (normals[scd_bnd,:]*area[scd_bnd,None]*g_ij[scd_bnd,None]).sum(0)
            F_bnd = F_bnd_fst + F_bnd_scd

            # Positional force

            fst_ij = fst & (lab[:,1]>=0)
            scd_ji = scd & (lab[:,0]>=0)

            d_ij = torch.maximum(torch.norm(simu.x[i,:] - simu.x[lab[fst_ij,1].int(),:],dim=1),torch.tensor(0.01))
            F_pos_ij = (-normals[fst_ij,:]*area[fst_ij,None]*1.0/d_ij[:,None]*b_ij[fst_ij,None]).sum(0)

            d_ji = torch.maximum(torch.norm(simu.x[i,:] - simu.x[lab[scd_ji,0].int(),:],dim=1),torch.tensor(0.01))
            F_pos_ji = (normals[scd_ji,:]*area[scd_ji,None]*1.0/d_ji[:,None]*b_ij[scd_ji,None]).sum(0)
            F_pos = F_pos_ij + F_pos_ji
            
            F[i,:] = F_pos + F_crv + F_bnd
        
        return F 
        
    def neigh_list_to_cc(neigh_list):
        cc = []
        cc_index = torch.zeros(N,dtype=neigh_list.dtype)
        seen = torch.zeros(N,dtype=bool)

        def add(b,index,cc,cc_index,seen):
            cc[index].append(b)
            seen[b] = True
            cc_index[b] = index

        def merge(i,j,cc,cc_index):
            cc[i] = cc[i] + cc[j]
            cc.pop(j)
            for index,component in enumerate(cc): 
                cc_index[component] = index

        for edge in neigh_list:
            a = edge[0].item()
            b = edge[1].item()
            if seen[a] & (~seen[b]):
                add(b,cc_index[a],cc,cc_index,seen)
            elif seen[b] & (~seen[a]):
                add(a,cc_index[b],cc,cc_index,seen)
            elif (~seen[a]) & (~seen[b]):
                cc.append([a,b])
                cc_index[a] = len(cc) - 1
                cc_index[b] = len(cc) - 1
                seen[a] = True
                seen[b] = True
            else:
                if (cc_index[a] != cc_index[b]):
                    merge(cc_index[a],cc_index[b],cc,cc_index)
                    
        return cc, cc_index

    def correction_force(F,lab):
        F_correction = torch.zeros_like(F)
        only_particles = (lab[:,0]>=0) & (lab[:,1]>=0)
        lab_particles = lab[only_particles,:]
        neigh_list = torch.unique(lab_particles,dim=0).to(device=lab.device,dtype=torch.long)
        cc, _ = neigh_list_to_cc(neigh_list)
        for component in cc:
            F_correction[component,:] = -F[component,:].mean(dim=0)[None,:]
        return F_correction
        

    #======================= INITIALISE ========================#

    solver.solve(simu,
                sinkhorn_algo=ot_algo,
                tau=0.4,
                to_bary=True,
                show_progress=False,
                default_init=False,
                weight=1.0,
                bsr=True)

    img = simu.labels.reshape(M,M,M).cpu().numpy()
    img[img==img.max()] = -1.0
    surfaces = compute_mesh(img)

    surfaces.save(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.vtk")
    img = np.pad(img,1,mode='constant',constant_values=-2.0)
    tif.imwrite(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.tif", img, bigtiff=True)

    solver.cost_params["C"] = sc

    t += dt
    t_iter += 1
    t_plot += 1

    #=========================== RUN ===========================#

    while t<T:
        print("--------------------------",flush=True)
        print(f"t={t}",flush=True)
        print("--------------------------",flush=True)

        plotting_time = t_iter%plot_every==0

        if plotting_time:
            print("I plot.",flush=True)
            solver.n_sinkhorn_last = 2000
            solver.n_sinkhorn = 2000
            solver.s0 = 2.0
        else:
            print("I do not plot.",flush=True)
            solver.n_sinkhorn_last = 250
            solver.n_sinkhorn = 250
            solver.s0 = 2*simu.R_mean
            
        
        F_inc = solver.lloyd_step(simu,
                    sinkhorn_algo=OT.LBFGSB,
                    tau=tau/(radius(simu)**(simu.d - 1)),
                    to_bary=False,
                    show_progress=False,
                    default_init=False,bsr=True)
        
        img = simu.labels.reshape(M,M,M).cpu().numpy()
        img[img==img.max()] = -1.0
        surfaces = compute_mesh(img)
        
        stime = time.time()
        stuff = extract_stuff(surfaces)
        print(f"Mesh extraction time: {time.time()-stime}",flush=True)
        
        stime = time.time()
        F_att = compute_forces(simu,*stuff)
        print(f"Force computation time: {time.time()-stime}",flush=True)
        
        stime = time.time()
        F_correct = correction_force(F_att,stuff[1])
        # F_correct = torch.tensor([[0.0,0.0,0.0]])
        print(f"Correction force computation time: {time.time()-stime}",flush=True)
        
        simu.x += F_att*dt + F_inc*dt + F_correct*dt
        
        print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}",flush=True)
        print(f"Maximal attraction force: {torch.max(torch.norm(F_att,dim=1))}",flush=True)
        print(f"Mean attraction force: {torch.mean(torch.norm(F_att,dim=1))}",flush=True)
        print(f"Maximal correction force: {torch.max(torch.norm(F_correct,dim=1))}",flush=True)
        print(f"Mean correction force: {torch.mean(torch.norm(F_correct,dim=1))}",flush=True)
        print(f"Maximal force: {torch.max(torch.norm(F_att + F_inc + F_correct,dim=1))}",flush=True)
        print(f"Mean force: {torch.mean(torch.norm(F_att + F_inc + F_correct,dim=1))}",flush=True)

        
        if plotting_time:
            if t_plot%save_every==0:
                surfaces.save(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.vtk")
                img = np.pad(img,1,mode='constant',constant_values=-2.0)
                # tif.imwrite(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.tif", img, bigtiff=True)
            t_plot += 1

        t += dt
        t_iter += 1
    
    t_plot +=1 
    surfaces.save(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.vtk")
    img = np.pad(img,1,mode='constant',constant_values=-2.0)
    tif.imwrite(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.tif", img, bigtiff=True)
    with open(simu_name + "/simu_final.pkl",'wb') as file:
        pickle.dump(simu,file)

def ratio_to_stparams(r_b1,r_g1,r_k1,g20=10.0):
    params = {
        "r_b1" : r_b1,
        "r_g1" : r_g1,
        "r_k1" : r_k1,
        "g20" : g20
    }
    
    k20 = 0.4
    k12 = k20
    params["gb"] = g20
    params["g11"] = 0.0
    params["g22"] = 0.0
    
    k10 = r_k1 * k20 
    params["g10"] = r_g1 * g20 
    
    params["b11"] = 2 * k10 * r_g1 * g20
    params["b22"] = 2 * k20 * g20
    params["b12"] = 2 * k10 * r_g1 * r_b1 * g20
    params["g12"] = k10/k12 * r_g1 * g20 
    return params


r_b1 = 3.0
r_g1 = 1.0
r_k1 = 1.0
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="separation")


r_b1 = 0.3
r_g1 = 1.0
r_k1 = 1.0
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="checkerboard")


r_b1 = 3.0
r_g1 = 2.0
r_k1 = 0.8/(r_b1*r_g1)
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="engulfment_bgeq1")

r_b1 = 0.3
r_g1 = 2.0
r_k1 = 0.8/r_g1
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="engulfment_bleq1")


r_b1 = 3.0
r_g1 = 2.0
r_k1 = 0.8/(r_b1*r_g1)
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="engulfment_bgeq1_split",init="split")

r_b1 = 0.3
r_g1 = 2.0
r_k1 = 0.8/r_g1
run_simu(ratio_to_stparams(r_b1,r_g1,r_k1,g20=5.0),title="engulfment_bleq1_split",init="split")