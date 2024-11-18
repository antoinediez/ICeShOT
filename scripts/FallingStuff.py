import os 
import sys
sys.path.append("..")
import time
import math
import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from pykeops.torch import LazyTensor
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

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"


simu_name = "simu_FallingStuff"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 50
N1 = N
N2 = N - N1
M = 512
dim = 3

vol_grid_true = 1.0/(M**dim)

cut = 0.03
o_cnt = 0.5 * torch.ones((1,dim))
o_cnt[:,-1] = 0.37
R_o = 0.1
    
def crop_function(x):
    cnt = 0.5 * torch.ones((1,dim))
    xc = x - cnt
    upper_cone = (xc[:,-1]>cut).float() * ((xc[:,:-1]**2).sum(1)<xc[:,-1]**2).float()
    below = (xc[:,-1]<cut).float()
    obstacle = (((x - o_cnt)**2).sum(1) > R_o**2).float()
    return upper_cone + below*obstacle

cnt_seeds = 0.5*torch.ones((1,dim))
size_seeds = 0.3
cnt_seeds[:,-1] = 1.0 - size_seeds/2

seeds = size_seeds*(torch.rand((N,dim))-0.5) + cnt_seeds
# source = sample.sample_cropped_domain(crop_function,n=M,dim=dim)
img = sample.sample_grid(M,dim=dim)
real_points = crop_function(img)>0
source = img[real_points,:]
all_labels = -1.0*torch.ones(M**dim)

N_max = 2000
vol0 = 0.5/N_max
R0 = math.sqrt(vol0/math.pi) if dim==2 else (vol0/(4./3.*math.pi))**(1./3.)
vol_x = vol0*torch.ones(N)

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void"
)

print(f"Number of voxels per particle: {int(vol0/simu.vol_grid)}")


p0 = 2
cost_params = {
    "p" : p0,
    "scaling" : "volume",
    "R" : R0,
    "C" : 1.0/(p0+2)
}

solver = OT_solver(
    n_sinkhorn=100,n_sinkhorn_last=100,n_lloyds=5,
    cost_function=costs.power_cost,cost_params=cost_params
)

T = 100.0
dt = 0.005
plot_every = 2
t = 0.0
t_iter = 0
t_plot = 0
F = torch.zeros((1,dim))
# F[0,-1] = -0.5
# tau = 3.0/R0 if dim==2 else 3.0/(R0**2)
F[0,-1] = -0.3
tau = 6.0/R0 if dim==2 else 6.0/(R0**2)

F0_ifc = 0.07

g11 = 1.0
g10 = 3.0

vol_cone = ((simu.y[:,-1]>0.5+cut).sum()*simu.vol_grid).item()

#==================== Plot config ======================#
pv.global_theme.volume_mapper = 'fixed_point'
pv.global_theme.render_lines_as_tubes = True

N = simu.N_cells
newcolors_all = np.zeros((N_max+1, 4))
cmap0 = plt.cm.hsv

for n in range(N_max+1):
    # newcolors_all[n,:3] = 0.1 + 0.8*np.random.rand(3)
    newcolors_all[n,:] = np.array(cmap0(n/N_max))
    newcolors_all[n,3] = 1.0
newcolors_all[0,:] = np.array([0.0,1.0,0.0,0.0])

cmap = ListedColormap(newcolors_all)
lut = pv.LookupTable(ListedColormap(newcolors_all),scalar_range=(-1.5,N_max-0.5))

def plot_cells(p,img,cmap="tab20b",**kwargs):
    img = np.pad(img,1,mode='constant',constant_values=-1.0)
    img[0,0,0] = N_max-1    # PyVista developers should be in prison for that
    p.add_volume(img,opacity='foreground',cmap=cmap,**kwargs)

box = pv.Cube(center=(M/2,M/2,M/2),x_length=M+2,y_length=M+2,z_length=M+2)
cone = pv.Cone(center=(M/2,M/2,0.75*M),direction=(0,0,-1),height=M/2,angle=45,resolution=100)
clipped_cone = cone.clip_closed_surface(normal=[0, 0, 1],origin=(M/2,M/2,M*(0.5 + cut)))
sphere = pv.Sphere(radius=R_o*M,center=(o_cnt[0,0].item()*M,o_cnt[0,1].item()*M,o_cnt[0,2].item()*M))
plane = pv.Plane(center=(M/2,M/2,M*(0.5 + cut)),direction=(0,0,1),i_size=M,j_size=M)


off_screen = True
plotter = pv.Plotter(lighting='three lights', off_screen=off_screen, image_scale=2)

#======================================================#


def sample_unit(N,d):
    x = torch.randn((N,d))
    x /= torch.norm(x,dim=1).reshape((N,1))
    return x

def insert(simu,n):
    simu.x = torch.cat((simu.x,size_seeds*(torch.rand((n,dim))-0.5) + cnt_seeds),dim=0)
    simu.x[-1,-1] = 0.95
    simu.axis = torch.cat((simu.axis,sample_unit(n,simu.d)),dim=0)
    simu.ar =torch.cat((simu.ar,torch.tensor([1.0])))
    simu.orientation = simu.orientation_from_axis()
    simu.N_cells += 1
    vol_particles = torch.cat((simu.volumes[:-1],torch.tensor([vol0])))
    simu.volumes = torch.cat((vol_particles,torch.tensor([1.0-vol_particles.sum()])))
    simu.f_x = torch.cat((torch.cat((simu.f_x[:-1],torch.tensor([0.0]))),torch.tensor([simu.f_x[-1]])))
    

#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=OT.LBFGSB,
             tau=1.0,
             to_bary=True,
             show_progress=False,
             bsr=True,
             weight=1.0)

N = simu.N_cells
simu.labels[simu.labels==torch.max(simu.labels)] = -1.0
all_labels[real_points] = simu.labels
all_labels[~real_points] = -1.0
img = all_labels.reshape(M,M,M).cpu().numpy() 
plot_cells(plotter,img,shade=True,diffuse=0.85,cmap=lut)
plotter.add_mesh(box, color='k', style='wireframe', line_width=3)
plotter.remove_scalar_bar()
plotter.add_mesh(clipped_cone,color='k',style='wireframe', line_width=1)
plotter.add_mesh(sphere)
plotter.add_mesh(plane,color='k',style='wireframe', line_width=1)
plotter.show(
    interactive=False,
    screenshot=simu_name + f'/frames/t_{t_plot}.png',
    cpos=[(2.4*M, 2.4*M, 1.2*M),(M/2, M/2, M/2),(0.0, 0.0, 1.0)],
    return_viewer=False,
    auto_close=False
)
plotter.clear_actors()
    
t += dt
t_iter += 1
t_plot += 1

solver.n_lloyds = 1
solver.cost_params["p"] = 10.0

with open(simu_name + f"/params.pkl",'wb') as file:
    pickle.dump(solver,file)

#=========================== RUN ===========================#

while t<T:
    print("--------------------------",flush=True)
    print(f"t={t}",flush=True)
    print("--------------------------",flush=True)
    
    plotting_time = t_iter%plot_every==0
    
    if plotting_time:
        print("I plot.",flush=True)
        solver.n_sinkhorn_last = 100
        solver.n_sinkhorn = 100
        solver.s0 = 2.0
        
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 100
        solver.n_sinkhorn = 100
        solver.s0 = 2.3*simu.R_mean
        
    # if simu.N_cells < N_max:
    #     if (simu.x[:,-1]>0.5+cut).sum()*vol0 < 0.85*vol_cone:
    #         if torch.rand(1)>0.42:
    #             insert(simu,1)
    #             print("+1",flush=True)
    #             print(f"N = {simu.N_cells}",flush=True)
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=OT.LBFGSB,
                tau=tau,
                to_bary=False,
                show_progress=False,
                default_init=False,bsr=True)

    simu.x += F*dt + F_inc*dt
    
    print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}",flush=True)
    
    if plotting_time:
        N = simu.N_cells
        simu.labels[simu.labels==torch.max(simu.labels)] = -1.0
        all_labels[real_points] = simu.labels
        all_labels[~real_points] = -1.0
        img = all_labels.reshape(M,M,M).cpu().numpy()   
        plot_cells(plotter,img,shade=True,diffuse=0.85,cmap=lut)
        plotter.add_mesh(box, color='k', style='wireframe', line_width=3)
        plotter.remove_scalar_bar()
        plotter.add_mesh(clipped_cone,color='k',style='wireframe', line_width=1)
        plotter.add_mesh(sphere)
        plotter.add_mesh(plane,color='k',style='wireframe', line_width=1)
        plotter.show(
            interactive=False,
            screenshot=simu_name + f'/frames/t_{t_plot}.png',
            cpos=[(2.4*M, 2.4*M, 1.2*M),(M/2, M/2, M/2),(0.0, 0.0, 1.0)],
            return_viewer=False,
            auto_close=False
        )
        plotter.clear_actors()
        t_plot += 1
    

    t += dt
    t_iter += 1
    
    if simu.N_cells<=N_max-1:
        T += dt
    
utils.make_video(simu_name=simu_name,video_name=simu_name)