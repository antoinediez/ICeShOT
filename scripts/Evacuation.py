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

p = 8

simu_name = "simu_Evacuation" + "_" + str(p)
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

base_color = colors.to_rgb('tab:blue')
cmap = utils.cmap_from_list(1000,0,0,color_names=["tab:blue","tab:orange","tab:gray"])

N = 111
M = 512

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
vol_x = 0.42*torch.ones(N)/N

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
    n_sinkhorn=800,n_sinkhorn_last=2000,n_lloyds=10,s0=2.0,
    cost_function=costs.power_cost,cost_params=cost_params
)

T = 15.0
# T = 5.0
dt = 0.001
plot_every = 5
t = 0.0
t_iter = 0
t_plot = 0
v0 = 0.4
diff = 0.2
tau = 3.0/simu.R_mean
# cap = 2**(p-1)
cap = None

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
    simu.labels[torch.isin(simu.labels,torch.where(who)[0])] = simu.x.shape[0] + 42
    
exit = torch.tensor([[1.0,0.5]])
    
simu.axis = (exit - simu.x)/torch.norm(exit - simu.x,dim=1).reshape((simu.N_cells,1))

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

quiv = simu_plot.ax.quiver(M*simu.x[:simu.N_cells,0].cpu(),M*simu.x[:simu.N_cells,1].cpu(),simu.axis[:simu.N_cells,0].cpu(),simu.axis[:simu.N_cells,1].cpu(),color='r',pivot='tail',zorder=2.5)
explot = simu_plot.ax.scatter(M*exit[:,0].cpu(),M*exit[:,1].cpu(),s=60,c='r',zorder=2.5)

simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")

quiv.remove()

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
    
    F_evacuation = (exit - simu.x)/torch.norm(exit - simu.x,dim=1).reshape((simu.N_cells,1))
    
    inRd =  F_evacuation*dt + math.sqrt(2*diff*dt)*torch.randn((simu.N_cells,2))
    simu.axis += inRd - (simu.axis * inRd).sum(1).reshape((simu.N_cells,1)) * simu.axis
    simu.axis /= torch.norm(simu.axis,dim=1).reshape((simu.N_cells,1))
    simu.x += v0*simu.axis*dt + F_inc*dt
    
    print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}")
    print(f"Average force: {torch.norm(v0*F_evacuation + F_inc,dim=1).mean()}")
    
    kill_index = (simu.x[:,0]>1.0-1.02*simu.R_mean) & (simu.x[:,1] < 0.5+1.05*simu.R_mean) & (simu.x[:,1] > 0.5-1.05*simu.R_mean)
    print(f"Exit: {kill_index.sum().item()}")
    
    kill(simu,kill_index)
    
    if plotting_time:
        simu_plot.update_plot(simu)
        quiv = simu_plot.ax.quiver(M*simu.x[:simu.N_cells,0].cpu(),M*simu.x[:simu.N_cells,1].cpu(),simu.axis[:simu.N_cells,0].cpu(),simu.axis[:simu.N_cells,1].cpu(),color='r',pivot='tail',zorder=2.5)
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        quiv.remove()
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)
        t_plot += 1

    t += dt
    t_iter += 1
    

utils.make_video(simu_name=simu_name,video_name=simu_name)