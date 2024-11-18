"""
Run-and-tumble in 2D with soft spheres in a disk domain
==========================================================

.. video:: ../../_static/SMV2_RunTumbleDisk.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

""" 

# sphinx_gallery_thumbnail_path = '_static/RunTumbleDisk.png'


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

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
    
# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB
    
simu_name = "simu_RunTumbleDisk"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 126
M = 512 

cnt = torch.tensor([[0.5,0.5]])
r_sq = 0.45**2 * torch.rand((N,1))
th = 2*math.pi * torch.rand((N,1))

seeds = cnt + torch.sqrt(r_sq) * torch.cat((torch.cos(th),torch.sin(th)),dim=1)

def disk(x):
    return ((0.5**2 - ((x - torch.tensor([[0.5,0.5]]))**2).sum(1))>0.0).float()

# source = sample.sample_grid(M)
source = sample.sample_cropped_domain(disk,M)
vol_x = torch.ones(N)
vol_x *= 0.875/vol_x.sum()

R0 = math.sqrt(vol_x[-1].item()/math.pi)

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc=None
)

eng = torch.linspace(0.4,3.5,N)

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
    "R" : math.sqrt(simu.volumes[0].item()/math.pi),
    "C" : 1.0
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=3000,n_lloyds=5,s0=2.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

cap = None

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False)


cost_params = {
    "scaling" : "volume",
    "R" : math.sqrt(simu.volumes[0].item()/math.pi),
    "C" : eng
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=3000,n_lloyds=5,s0=1.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

cap = None

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False,
             default_init=False)

# cmap = plt.cm.bone_r
# cmap = colors.LinearSegmentedColormap.from_list("trucated",cmap(np.linspace(0.1, 0.9, 100)))
clrs = [colors.to_rgb('w'), colors.to_rgb('xkcd:prussian blue')] # first color is black, last is red
cmap = colors.LinearSegmentedColormap.from_list(
        "Custom", clrs, N=1000)

cmap = colors.LinearSegmentedColormap.from_list("trucated",cmap(np.linspace(0.2, 1.0, 1000)))

simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
                 plot_pixels=True,plot_scat=True,plot_quiv=True,plot_boundary=True,
                 scat_size=15,scat_color='k',
                 r=None,K=5,boundary_color='k',
                 plot_type="scatter",void_color='w',M_grid=M)

alp = np.zeros(N)
alp[1] = 1.0
alp[-1] = 1.0
alp[int(0.25*N)] = 1.0
alp[int(0.5*N)] = 1.0
alp[int(0.75*N)] = 1.0

simu_plot.plots["quiv"].set(alpha=alp)


simu_plot.ax.plot(M*(0.5+0.5*np.cos(2*math.pi*np.linspace(0,1,100))),M*(0.5+0.5*np.sin(2*math.pi*np.linspace(0,1,100))),color='k',linewidth=3.0)

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
        
    F_inc = solver.lloyd_step(simu,
            sinkhorn_algo=ot_algo,cap=cap,
            tau=1.0/simu.R_mean,
            to_bary=False,
            show_progress=False,
            default_init=di)
    
    simu.x +=  v0*simu.axis*dt + Finc0*F_inc*dt
    
    who_jumps = torch.rand(N) > math.exp(-jump_rate*dt)
    simu.axis[who_jumps,:] = torch.randn((who_jumps.sum(),2))
    simu.axis[who_jumps,:] /= torch.norm(simu.axis[who_jumps,:],dim=1).reshape((who_jumps.sum(),1))
    
    if plotting_time:
        simu_plot.update_plot(simu)
        simu_plot.fig
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)

        t_plot += 1
    
    t += dt
    t_iter += 1
    
utils.make_video(simu_name=simu_name,video_name=simu_name)
