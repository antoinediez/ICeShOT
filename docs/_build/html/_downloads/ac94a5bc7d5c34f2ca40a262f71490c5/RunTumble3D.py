"""
Run-and-tumble in 3D with soft spheres
============================================

.. video:: ../_static/SMV25_RunTumble_3D.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

""" 

# sphinx_gallery_thumbnail_path = '_static/tumble_bricks.png'


import os 
import sys
sys.path.append("..")
import pickle
import math
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from iceshot import cells
from iceshot import costs
from iceshot import OT
from iceshot.OT import OT_solver
from iceshot import plot_cells
from iceshot import sample
from iceshot import utils
from iceshot.cells import DataPoints
from pykeops.torch import LazyTensor
from tqdm import tqdm 
import tifffile as tif

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
    
# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB
    
simu_name = "simu_RunTumble3D"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 126
M = 128 

source = sample.sample_grid(M,dim=3)
seeds = torch.rand(N,3)

vol_x = 0.5 + 0.5*torch.rand(N)
vol_x *= 0.8/vol_x.sum()

R0 = (vol_x[-1].item()/(4./3.*math.pi)) ** (1./3.)

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc=None
)

eng = torch.linspace(0.5,3.5,N)

T = 10.0
dt = 0.0025
plot_every = 4
t = 0.0
t_iter = 0
t_plot = 0

Finc0 = 0.2
jump_rate = 1.0
v0 = 0.4


#======================= INITIALISE ========================#


cost_params = {
    "scaling" : "volume",
    "R" : R0,
    "C" : 1.0
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=2000,n_lloyds=5,s0=2.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

cap = None

solver.solve(simu,
             sinkhorn_algo=OT.sinkhorn_zerolast,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False)


cost_params = {
    "scaling" : "volume",
    "R" : R0,
    "C" : eng
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=2000,n_lloyds=5,s0=1.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

cap = None

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False,
             default_init=False)

tif.imsave(simu_name + "/frames/"+f"t_{t_iter}.tif", simu.labels.reshape(M,M,M).cpu().numpy(), bigtiff=True)
t_plot += 1
t += dt
t_iter += 1

#=========================== RUN ===========================#

while t<T:
    print("--------------------------",flush=True)
    print(f"t={t}",flush=True)
    print("--------------------------",flush=True)

    plotting_time = t_iter%plot_every==0
    
    if plotting_time:
        print("I plot.",flush=True)
        solver.n_sinkhorn_last = 3000
        solver.n_sinkhorn = 3000
        solver.s0 = 1.5
        di = False
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 300
        solver.n_sinkhorn = 300
        solver.s0 = 2*simu.R_mean
        di = False
        
    R = (simu.volumes[:-1]/(4./3.*math.pi)) ** (1./3.)
        
    F_inc = solver.lloyd_step(simu,
            sinkhorn_algo=ot_algo,cap=cap,
            tau=1.0/(R ** 2),
            to_bary=False,
            show_progress=False,
            default_init=di)
    
    simu.x +=  v0*simu.axis*dt + Finc0*F_inc*dt
    
    who_jumps = torch.rand(N) > math.exp(-jump_rate*dt)
    simu.axis[who_jumps,:] = torch.randn((who_jumps.sum(),3))
    simu.axis[who_jumps,:] /= torch.norm(simu.axis[who_jumps,:],dim=1).reshape((who_jumps.sum(),1))
    
    if plotting_time:
        tif.imsave(simu_name + "/frames/"+f"t_{t_plot}.tif", simu.labels.reshape(M,M,M).cpu().numpy(), bigtiff=True)
        t_plot += 1
    
    t += dt
    t_iter += 1