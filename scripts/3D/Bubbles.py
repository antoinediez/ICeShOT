"""
Bubbles
============================================

This is a non-scientific experiment with bubbles. 

Bubbles are generated randomly within the lower half of the domain (fluid) where they are subject to a buoyancy force. In the upper half of the domain (air) they are subject to gravity so the bubbles are constrained to stay at the surface. Bubbles can explode (with a higher probability when they are big) and fuse. Otherwise, they interact according to the force describe in the 3D cell sorting experiment with all parameters equal in order to ensure the (first two) Plateau conditions. 

.. video:: ../../_static/SMV23_Bubbles.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400

"""

# sphinx_gallery_thumbnail_path = '_static/bubbles.png'


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
    
simu_name = "simu_foam"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB

def radius(simu):
    return torch.sqrt(simu.volumes[:-1]/math.pi) if simu.d==2 else (simu.volumes[:-1]/(4./3.*math.pi)) ** (1./3.)

N = 42
M = 256
dim = 3

R00 = 1.0
vol_max = 0.25
R0 = 0.021
vol0 = math.pi*(R0**2) if dim==2 else 4./3.*math.pi*(R0**3)

seeds = torch.rand((N,dim))
seeds[:,-1] *= 0.4
source = sample.sample_grid(M,dim=dim)
vol_x = vol0*(1.0 + 4*torch.rand(N))

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",jct_method='Kmin'
)

print(f"Number of pixels min: {vol0/simu.vol_grid}")
if vol0/simu.vol_grid < 200:
    raise ValueError("Not enough pixels")

cost_params = {
    "p" : 2,
    "scaling" : "constant",
    "C" : R00/radius(simu)
}

solver = OT_solver(
    n_sinkhorn=800,n_sinkhorn_last=2000,n_lloyds=10,s0=2.0,
    cost_function=costs.power_cost,cost_params=cost_params
)

T = 120.0
dt = 0.0025
plot_every = 40
save_every = 1
t = 0.0
t_iter = 0
t_plot = 0

g12 = 0.7
g11 = 0.7
g22 = 0.7
g02 = 0.7
g01 = 0.7
gb = 5.0

b12 = 0.7
b11 = 0.7
b22 = 0.7

tau  = 0.0

pop_rate = 120.0
scale_prob_fuse = 6.0
dying_rate = 85.0

#===========================================================#



def compute_mesh(img):
    img = np.pad(img,1,mode='constant',constant_values=-2.0)
    vol = pv.wrap(img)
    alg = vtk.vtkSurfaceNets3D()
    # alg = vtk.vtkDiscreteFlyingEdges3D()
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
    
    
    g_ij[(lab[:,0]>=0) & (lab[:,1]>=0)] = g12

    g_ij[(lab[:,0]==-1) & (lab[:,1]>=0)] = g01
    g_ij[(lab[:,1]==-1) & (lab[:,0]>=0)] = g01

    g_ij[(lab[:,0]==-2) | (lab[:,1]==-2)] = gb


    b_ij = torch.zeros(len(lab))
    b_ij[(lab[:,0]>=0) & (lab[:,1]>=0)] = b12
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
    
def neigh_list_to_cc(neigh_list,N):
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
    print(N,flush=True)
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
    cc, _ = neigh_list_to_cc(neigh_list,N=len(F))
    for component in cc:
        F_correction[component,:] = -F[component,:].mean(dim=0)[None,:]
    return F_correction
    
def boundary_attraction(simu,F_bnd_att=0.015):
    F = torch.zeros_like(simu.x)
    surf = (((simu.x[:,-1]>0.45) & (simu.x[:,-1]<0.55))) | ((simu.x - 0.5).abs().max(dim=1).values > 0.22)
    touching = (simu.x - 0.5).abs().max(dim=1).values > 0.24
    F[(surf)&(~touching) ,:] = F_bnd_att * (simu.x[(surf)&(~touching),:] - 0.5)
    return F
    

def force(x,F_buo=0.5,F_gra=-0.15,r=2*R0):
    N,d = x.shape
    F = torch.clamp((F_gra - F_buo)/(2*r)*(x[:,-1] - 0.5),min=F_gra,max=F_buo)
    return torch.cat((torch.zeros((N,d-1)),F.reshape((N,1))),dim=1)

def sample_unit(N,d):
    x = torch.randn((N,d))
    x /= torch.norm(x,dim=1).reshape((N,1))
    return x

def insert(simu,n):
    new_x = torch.rand((n,dim))
    new_x[:,-1] *= 0.3
    simu.x = torch.cat((simu.x,new_x),dim=0)
    simu.axis = torch.cat((simu.axis,sample_unit(n,simu.d)),dim=0)
    simu.ar = torch.cat((simu.ar,torch.ones(n)))
    simu.orientation = simu.orientation_from_axis()
    simu.N_cells += n
    vol_particles = torch.cat((simu.volumes[:-1],vol0*torch.ones(n)))
    simu.volumes = torch.cat((vol_particles,torch.tensor([1.0-vol_particles.sum()])))
    simu.f_x = torch.cat((torch.cat((simu.f_x[:-1],torch.zeros(n))),torch.tensor([simu.f_x[-1]])))
    simu.labels[simu.labels==simu.labels.max()] = simu.x.shape[0] + 42

def kill(simu,who,solver=solver,cost_matrix=None):
    who_p = torch.cat((who,torch.zeros(1,dtype=bool,device=who.device)))
    simu.x = simu.x[~who]
    simu.f_x = simu.f_x[~who_p]
    simu.volumes[-1] += simu.volumes[who_p].sum()
    simu.volumes = simu.volumes[~who_p]
    simu.axis = simu.axis[~who]
    simu.ar = simu.ar[~who]
    simu.orientation = simu.orientation[~who]
    simu.N_cells -= int(who.sum().item())
    simu.labels[simu.labels==simu.labels.max()] = simu.x.shape[0] + 42
    simu.labels[torch.isin(simu.labels,torch.where(who)[0])] = simu.x.shape[0] + 42

def fusion(simu,ind1,ind2):
    N_ind = len(ind1)
    simu.x[ind1,:] = (simu.volumes[ind1].reshape((N_ind,1))*simu.x[ind1,:] + simu.volumes[ind2].reshape((N_ind,1))*simu.x[ind2,:])/(simu.volumes[ind1].reshape((N_ind,1)) + simu.volumes[ind2].reshape((N_ind,1)))
    simu.volumes[ind1] = simu.volumes[ind1] + simu.volumes[ind2]
    simu.labels[ind2] = simu.labels[ind1]
    who_kill = torch.zeros(simu.N_cells,dtype=torch.bool)
    who_kill[ind2] = True
    kill(simu,who_kill)
    
def filter_by_fusion_probability(unq_bnd,prob_fuse):
    _, ind_prob_fuse = torch.sort(prob_fuse,descending=True)
    sorted = unq_bnd[ind_prob_fuse,:]
    keep = sorted[0,:].reshape(1,2).detach().clone()
    filtered_prob = prob_fuse[0].reshape(1)
    for i in range(len(sorted)-1):
        if torch.isin(sorted[i+1,:],keep).sum() == 0.0:
            keep = torch.cat((keep,sorted[i+1,:].reshape(1,2)))
            filtered_prob =  torch.cat((filtered_prob,prob_fuse[i+1].reshape(1)))
    return keep,filtered_prob
    

#======================= 3D PLOTTING ========= =============#

box = pv.Cube(center=(M/2,M/2,M/2),x_length=M+2,y_length=M+2,z_length=M+2)

off_screen = True
plotter = pv.Plotter(off_screen=off_screen, image_scale=2)
plotter.enable_ssao(radius=15, bias=0.5)
plotter.enable_anti_aliasing("ssaa")
# cpos = [(0.5*M, -2*M, 0.5*M),(M/2, M/2, M/2),(0.0, 0.0, 1.0)]
plotter.camera_position = [(M/2,M/2,3*M), (M/2,M/2,M/2), (0,1,0)]
water = pv.Cube(center=(M/2,M/2,M/4),x_length=M+2,y_length=M+2,z_length=M/2)
cpos = [(2.4*M, 2.8*M, 2*M),(M/2, M/2, M/2),(0.0, 0.0, 1.0)]

#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=ot_algo,
             tau=0.0,
             to_bary=True,
             show_progress=False,
             default_init=False,
             weight=1.0,
             bsr=True)

plotter = pv.Plotter(lighting='three lights', image_scale=2)
img = simu.labels.reshape(M,M,M).cpu().numpy()
img[img==img.max()] = -1.0
# plot_cells(plotter,img,shade=True,diffuse=0.85)
surfaces = compute_mesh(img)
plotter.add_mesh(
        surfaces,
        interpolation="gouraud",
        roughness=1.0,
        ambient=0.8,
        diffuse=0.2,
        opacity=0.3,
        cmap=["tab:cyan"],
        specular=1.0,
        show_edges=True
    )
plotter.add_mesh(box, color='k', style='wireframe', line_width=1.2)
plotter.add_mesh(water,color='tab:cyan', opacity=0.2, specular=1.0,line_width=1.0,show_edges=True)
plotter.remove_scalar_bar()
plotter.show(
    interactive=False,
    screenshot=simu_name + f'/frames/t_{t_plot}.png',
    cpos=cpos,
    return_viewer=False,
    auto_close=False
)


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
        
    who_dies = (torch.rand(simu.N_cells) > torch.exp(-simu.volumes[:-1]*dt*dying_rate)) & (simu.x[:,-1] > 0.45)
    kill(simu,who_dies)
        
    y_ind, x_ind = simu.extract_boundary()
    
    medium_bnd = (x_ind == simu.N_cells + 42.0).sum(1)
    y_ind = y_ind[medium_bnd<0.1]
    x_ind = x_ind[medium_bnd<0.1,:]
    if len(x_ind)>0:
        unq_bnd, count = torch.unique(x_ind,dim=0,return_counts=True)
        prob_fuse = scale_prob_fuse/count * dt
        ind_to_fusion, filtered_prob = filter_by_fusion_probability(unq_bnd,prob_fuse)
        to_fuse = torch.rand(len(ind_to_fusion)) < filtered_prob
        fusion(simu,ind_to_fusion[to_fuse,0],ind_to_fusion[to_fuse,1])
        
    if simu.volumes[:-1].sum()<vol_max:
        n = np.random.poisson(pop_rate*dt)
    insert(simu,n)
    
    F_ext = force(simu.x)
    
    solver.cost_params["C"] = R00/radius(simu)  
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=OT.LBFGSB,
                tau=tau/(radius(simu)**(simu.d - 1)),
                to_bary=False,
                show_progress=False,
                default_init=False,bsr=True)
    
    img = simu.labels.reshape(M,M,M).cpu().numpy()
    img[img==img.max()] = -1.0
    surfaces = compute_mesh(img)
    stuff = extract_stuff(surfaces)
        
    F_att = compute_forces(simu,*stuff)
    F_correct = correction_force(F_att,stuff[1])
    F_bnd = boundary_attraction(simu)
    
    F_tot = F_att + F_inc + F_ext + F_bnd + F_correct
    
    torch.clamp(simu.x,0.02,0.98,out=simu.x)
    
    print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}",flush=True)
    print(f"Maximal attraction force: {torch.max(torch.norm(F_att,dim=1))}",flush=True)
    print(f"Maximal external force: {torch.max(torch.norm(F_ext,dim=1))}",flush=True)
    print(f"Maximal boundary force: {torch.max(torch.norm(F_bnd,dim=1))}",flush=True)
    print(f"Maximal force: {torch.max(torch.norm(F_tot,dim=1))}",flush=True)
    
    nan_check = torch.isnan(F_tot)
    print(f"Number of NaN: {nan_check.sum()}")
    F_tot[nan_check] = 0.0

    simu.x += F_tot*dt

    
    if plotting_time:
        # stime = time.time()
        # plotter.add_mesh(
        #         surfaces,
        #         interpolation="gouraud",
        #         roughness=1.0,
        #         ambient=0.8,
        #         diffuse=0.2,
        #         opacity=0.3,
        #         cmap=["tab:cyan"],
        #         specular=1.0,
        #         show_edges=True
        #     )
        # plotter.add_mesh(box, color='k', style='wireframe', line_width=1.2)
        # plotter.add_mesh(water,color='tab:cyan', opacity=0.2, specular=1.0,line_width=1.0,show_edges=True)
        # plotter.remove_scalar_bar()
        # plotter.show(
        #     interactive=False,
        #     screenshot=simu_name + f'/frames/t_{t_plot}.png',
        #     cpos=cpos,
        #     return_viewer=False,
        #     auto_close=False
        # )
        # plotter.clear_actors()
        # ptime = time.time() - stime
        # print(f"Plotting time: {ptime} seconds for {simu.N_cells} cells")
        if t_plot%save_every==0:
            surfaces.save(simu_name + "/data/"+f"t_{int(t_plot/save_every)}.vtk")
        t_plot += 1

    t += dt
    t_iter += 1

utils.make_video(simu_name=simu_name,video_name=simu_name)