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
    
simu_name = "simu_RodShape"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 300
M = 512

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
vol_x = 0.65*torch.ones(N)/N

ar = 3.0
simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    ar=ar,bc="periodic"
)

p = 3.5
cost_params = {
    "p" : p,
    "scaling" : "volume",
    "b" : math.sqrt(simu.volumes[0].item()/(math.pi + 4*(ar-1))),
    "C" : 1.0
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=3000,n_lloyds=20,s0=2.0,
    cost_function=costs.spherocylinders_2_cost,cost_params=cost_params
)

T = 30.0
dt = 0.002
plot_every = 4
t = 0.0
t_iter = 0
t_plot = 0
v0 = 0.3
tau = torch.ones(N)/simu.R_mean
tau *= 0.14
# tau = torch.ones(N)
# tau *= 10.0
cap = None

cmap = utils.cmap_from_list(N,0,0,color_names=["tab:blue","tab:blue","tab:blue"])

#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=OT.sinkhorn_zerolast,cap=cap,
             tau=1.0,
             to_bary=True,
             show_progress=False,weight=1.0)

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
        solver.n_sinkhorn_last = 3000
        solver.n_sinkhorn = 3000
        solver.s0 = 1.0
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 300
        solver.n_sinkhorn = 300
        solver.s0 = simu.R_mean
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=OT.sinkhorn_zerolast,cap=cap,
                tau=tau,
                to_bary=False,
                show_progress=False,
                default_init=False)
    
    simu.x += v0*simu.axis*dt + F_inc*dt
        
    cov = simu.covariance_matrix()
    cov /= torch.sqrt(torch.det(cov).reshape((simu.N_cells,1,1)))
    L,Q = torch.linalg.eigh(cov)
    axis = Q[:,:,-1]
    axis = (axis * simu.axis).sum(1).sign().reshape((simu.N_cells,1)) * axis
    simu.axis = axis
    simu.orientation = simu.orientation_from_axis()
    
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




