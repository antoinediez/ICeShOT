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

p = 2

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB

simu_name = "simu_TissueGrowth"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

cmap = utils.cmap_from_list(1000,color_names=["tab:blue"])

N = 1
M = 800
Nmax = 400
vol0 = 0.5*0.75/Nmax
vol1 = 0.75/Nmax
R1 = math.sqrt(vol1/math.pi)

seeds = torch.tensor([[0.5,0.5]])
source = sample.sample_grid(M)
vol_x = torch.tensor([vol1])

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

T = 30.0
# T = 5.0
dt = 0.01
plot_every = 1
t = 0.0
t_iter = 0
t_plot = 0
growth_rate = (vol1-vol0)/0.5
growth_rate_factor = 0.5 + 1.5*torch.rand(simu.N_cells)
div_rate = 5.0
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
    
exit = torch.tensor([[0.5,0.5]])
#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=0.0,
             to_bary=True,
             show_progress=False)

simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
                 plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=True,
                 scat_size=5,scat_color='k',
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
        
    simu.volumes[:-1] += growth_rate_factor * growth_rate*dt
    simu.volumes[:-1] = torch.minimum(simu.volumes[:-1],torch.tensor([vol1]))
    simu.volumes[-1] = 1.0 - simu.volumes[:-1].sum()
    
    who_divide = (simu.volumes[:-1] > 0.8*vol1) & (torch.rand(simu.N_cells) > math.exp(-dt*div_rate))
    
    for ind,who in enumerate(who_divide):
        if who:
            if simu.N_cells<=Nmax:
                divide(simu,ind,R1)
                growth_rate_factor = insert(growth_rate_factor,ind,growth_rate_factor[ind],0.5+1.5*torch.rand(1))
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=cap,
                tau=1.0/torch.sqrt(simu.volumes[:-1]/math.pi),
                to_bary=False,
                show_progress=False,
                default_init=False)
    
    F_evacuation = (exit - simu.x)/(torch.norm(exit - simu.x,dim=1).reshape((simu.N_cells,1)) + 1e-6)
    
    simu.x += F_inc*dt + 0.2*F_evacuation*dt
    
    try:
        cov = simu.covariance_matrix()
        cov /= torch.sqrt(torch.det(cov).reshape((simu.N_cells,1,1)))
        L,Q = torch.linalg.eigh(cov)
        axis = Q[:,:,-1]
        axis = (axis * simu.axis).sum(1).sign().reshape((simu.N_cells,1)) * axis
        simu.axis = axis
        simu.orientation = simu.orientation_from_axis()
    except:
        pass
    
    print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}")
        
    if plotting_time:
        simu_plot.update_plot(simu)
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)
        t_plot += 1

    t += dt
    t_iter += 1
    

utils.make_video(simu_name=simu_name,video_name=simu_name)