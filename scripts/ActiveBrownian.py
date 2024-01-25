import os 
import sys
sys.path.append("..")
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

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"

p = 2.5

simu_name = "simu_ActiveBrownian_p" + str(p)
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

base_color = colors.to_rgb('tab:blue')
cmap = utils.cmap_from_list(1000,0,0,color_names=["tab:blue","tab:orange","tab:gray"])

N = 250
M = 512

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
vol_x = 0.94*torch.ones(N)/N

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc="periodic"
)

cost_params = {
    "p" : p,
    "scaling" : "volume",
    "R" : simu.R_mean,
    "C" : 0.1
}

solver = OT_solver(
    n_sinkhorn=800,n_sinkhorn_last=2000,n_lloyds=10,s0=2.0,
    cost_function=costs.power_cost,cost_params=cost_params
)

# T = 12.0
T = 5.0
dt = 0.002
plot_every = 5
t = 0.0
t_iter = 0
t_plot = 0
v0 = 0.3
diff = 20.0
tau = torch.ones(N)/simu.R_mean
tau *= 3.0
# cap = 2**(p-1)
cap = None

#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=OT.sinkhorn_zerolast,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False)

simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
                 plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=True,
                 scat_size=15,scat_color='k',
                 r=None,K=5,boundary_color='k',
                 plot_type="imshow",void_color='w')

simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")

with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
    pickle.dump(simu,file)
    
t += dt
t_iter += 1
t_plot += 1

solver.n_lloyds = 1
solver.cost_params["p"] = p

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
        solver.n_sinkhorn_last = 2000
        solver.n_sinkhorn = 2000
        solver.s0 = 2.0
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 250
        solver.n_sinkhorn = 250
        solver.s0 = 2*simu.R_mean
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=OT.sinkhorn_zerolast,cap=cap,
                tau=tau,
                to_bary=False,
                show_progress=False,
                default_init=False)
    
    simu.x += v0*simu.axis*dt
    
    simu.axis += math.sqrt(2*diff*dt)*torch.randn((N,2))
    simu.axis /= torch.norm(simu.axis,dim=1).reshape((N,1))
    
    simu.x += F_inc*dt
    
    simu.x = torch.remainder(simu.x,1)

    print(torch.max(torch.norm(F_inc,dim=1)))
    
    if plotting_time:
        simu_plot.update_plot(simu)
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)
        t_plot += 1

    t += dt
    t_iter += 1
    
utils.make_video(simu_name=simu_name,video_name=simu_name)




