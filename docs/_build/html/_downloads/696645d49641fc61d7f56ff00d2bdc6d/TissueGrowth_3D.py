"""
Growth of a 3D cell aggregate
============================================

We consider a 3D cell aggregate growing according to a basic somatic cell cycle.
Starting from one cell, each cell grows at a linear speed until a target volume is reached, then it 
divides after a random exponential time producing two daughter cells with identical half volumes.

We keep a constant resolution throughout the simulation by progressibely zooming out as the aggregate grows. 

.. video:: ../_static/TissueGrowth_3D.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

""" 

# sphinx_gallery_thumbnail_path = '_static/tissue_growth_3D.png'


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


use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"

p = 2

ot_algo = OT.LBFGSB

simu_name = "simu_TissueGrowth_3D"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

Nmax = 50000
N = 1
M = 400
R1 = 0.15
vol1 = 4./3. * math.pi * (R1**3)
vol0 = 0.5*vol1
cnt = torch.tensor([[0.5,0.5,0.5]])

vol1_end = 1.0/Nmax
R1_end = (vol1_end/(4./3.*math.pi)) ** (1./3.)
scale = R1_end/R1

seeds = torch.tensor([[0.5,0.5,0.5]])
source = sample.sample_grid(M,dim=3)
vol_x = vol1*torch.ones(N)

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc=None
)

cost_params = {
    "p" : p,
    "scaling" : "volume",
    "R" : simu.R_mean,
    "C" : 0.1
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=1000,n_lloyds=4,s0=2.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

dt = 0.002
plot_every = 3
t = 0.0
t_iter = 0
t_plot = 0
growth_rate = 6.0*(vol1-vol0)
growth_rate_factor = 0.5 + 1.5*torch.rand(simu.N_cells)
div_rate = 3.0
cap = None

def insert(x,ind,elem1,elem2):
    sh = list(x.shape)
    sh[0] += 1
    new_x = torch.zeros(sh)
    new_x[:ind] = x[:ind]
    new_x[(ind+2):] = x[(ind+1):]
    new_x[ind] = elem1
    new_x[ind+1] = elem2
    return new_x

def sample_unit(N,d):
    x = torch.randn((N,d))
    x /= torch.norm(x,dim=1).reshape((N,1))
    return x

def divide(simu,ind,R1):
    simu.x = insert(simu.x,ind,simu.x[ind]-0.5*R1*simu.axis[ind],simu.x[ind]+0.5*R1*simu.axis[ind])
    simu.axis = insert(simu.axis,ind,sample_unit(1,simu.d),sample_unit(1,simu.d))
    simu.ar = insert(simu.ar,ind,1.0,1.0)
    simu.orientation = simu.orientation_from_axis()
    simu.N_cells += 1
    simu.volumes = insert(simu.volumes,ind,0.5*simu.volumes[ind],0.5*simu.volumes[ind])
    simu.f_x = insert(simu.f_x,ind,simu.f_x[ind],simu.f_x[ind])

total_vol = simu.volumes[:-1].sum().item()

data = {
    "N" : [1],
    "T" : [0.0],
    "vol" : [total_vol],
    "scale" : [scale]
}
#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=0.7,
             to_bary=True,
             show_progress=False,
             default_init=False,
             stopping_criterion="average",
             tol=0.01)
    
t += dt
t_iter += 1
t_plot += 1

solver.n_lloyds = 1
solver.cost_params["p"] = p

#=========================== RUN ===========================#

stime = time.time()

while True:
    print("--------------------------",flush=True)
    print(f"t={t}",flush=True)
    print(f"N={simu.N_cells}",flush=True)
    print(f"V={total_vol}",flush=True)
    print("--------------------------",flush=True)
    
    plotting_time = t_iter%plot_every==0
    
    if plotting_time:
        print("I plot.",flush=True)
        solver.n_sinkhorn_last = 200
        solver.n_sinkhorn = 200
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 200
        solver.n_sinkhorn = 200
        
    simu.volumes[:-1] += growth_rate_factor * growth_rate*dt
    simu.volumes[:-1] = torch.minimum(simu.volumes[:-1],torch.tensor([vol1]))
    simu.volumes[-1] = 1.0 - simu.volumes[:-1].sum()
    
    who_divide = (simu.volumes[:-1] > 0.8*vol1) & (torch.rand(simu.N_cells) > math.exp(-dt*div_rate)) & (torch.max(torch.abs(simu.x - cnt),dim=1)[0] < 0.5 - R1) 
    
    for ind,who in enumerate(who_divide):
        if who:
            if simu.N_cells<=Nmax:
                divide(simu,ind,R1)
                growth_rate_factor = insert(growth_rate_factor,ind,growth_rate_factor[ind],0.5+1.5*torch.rand(1))
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=cap,
                tau=10.0/(R1**2),
                to_bary=False,
                show_progress=False,
                default_init=False,
                stopping_criterion="average",
                tol=0.01)
        
    simu.x += F_inc*dt
    
    print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}")
    
    simu.labels[simu.labels==torch.max(simu.labels)] = -100.0
    
    total_vol = simu.volumes[:-1].sum().item()
    R_m = 1.1 * (total_vol/(4./3.*math.pi)) ** (1./3.)
    ratio = min(1.0,0.3/R_m)
    print(f"RATIO={ratio}",flush=True)
    new_scale = min(1.0,1/ratio * scale)
    ratio = scale/new_scale
    simu.x = cnt + ratio*(simu.x - cnt)
    vol0 *= ratio**3
    vol1 *= ratio**3
    R1 *= ratio
    simu.R_mean *= ratio
    simu.volumes[:-1] *= ratio**3
    simu.volumes[-1] = 1.0 - simu.volumes[:-1].sum()
    scale = new_scale
    print(f"SCALE={scale}",flush=True)
        
    if plotting_time:        
        tif.imsave(simu_name + "/frames/"+f"t_{t_plot}.tif", simu.labels.reshape(M,M,M).cpu().numpy(), bigtiff=True)
        t_plot += 1
        data["N"].append(simu.N_cells)
        data["T"].append(time.time() - stime)
        data["vol"].append(total_vol)
        data["scale"].append(scale)
        pickle.dump(data,open(simu_name+"/data.p","wb"))
        if total_vol>0.9999 and simu.N_cells>Nmax:
            with open(simu_name + "/data/data_final.pkl",'wb') as file:
                pickle.dump(simu,file)
            break

    t += dt
    t_iter += 1