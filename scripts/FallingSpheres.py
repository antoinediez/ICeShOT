"""
Falling soft spheres in 2D
============================================


With :math:`p=0.75`

.. video:: ../_static/SMV4_FallingSpheres_p075.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`p=2`

.. video:: ../_static/SMV5_FallingSpheres_p2.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`p=10`

.. video:: ../_static/SMV6_FallingSpheres_p10.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|


With :math:`\\tau_o=8` and :math:`\\tau_b=3`

.. video:: ../_static/SMV7_FallingSpheres_tau8.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`\\tau_o=3` and :math:`\\tau_b=3`

.. video:: ../_static/SMV8_FallingSpheres_tau3.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`\\tau_o=1` and :math:`\\tau_b=3`

.. video:: ../_static/SMV9_FallingSpheres_tau1.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|


""" 

# sphinx_gallery_thumbnail_path = '_static/FallingSpheres_softheavy.png'

import os 
import sys
sys.path.append("..")
import pickle
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

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB

p_b = 10
tau_b = 1.5
p_o = 1
tau_o = 3

# simu_name = "simu_FallingSpheres" + "_p" + str(p_o) + "_tau" + str(tau_o) 
simu_name = "simu_FallingSpheres" + "_p" + str(p_b) + "_b"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")


N = 30
# N = 42
# N1 = 21
N1 = N
N2 = N - N1
M = 512
# M = 300

cmap = utils.cmap_from_list(N1,N2,0,color_names=["tab:blue","tab:orange","tab:gray"])

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
# vol_x = 0.5*torch.ones(N)/N
vol_x = 0.3*torch.ones(N)/N

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void"
)

p = torch.ones(N)
p[:N1] = p_b
p[N1:] = p_o
p0 = 6
cost_params = {
    "p" : p0,
    "scaling" : "volume",
    "R" : simu.R_mean,
    "C" : 1.0/(p0+2)
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=3000,n_lloyds=14,
    cost_function=costs.power_cost,cost_params=cost_params
)

T = 10.0
dt = 0.001
plot_every = 5
t = 0.0
t_iter = 0
t_plot = 0
F = torch.tensor([[0.0,-0.4]])
# F = torch.tensor([[0.0,-0.25]])
tau = torch.ones(N)/simu.R_mean
tau[:N1] *= tau_b
# tau[:N1] *= 1.0
tau[N1:] *= tau_o
# cap = 2**(p0-1)
cap = None

#======================= INITIALISE ========================#

solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
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
        solver.n_sinkhorn_last = 3000
        solver.n_sinkhorn = 3000
        solver.s0 = 2.0
        
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 400
        solver.n_sinkhorn = 400
        solver.s0 = 2.3*simu.R_mean
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=cap,
                tau=tau,
                to_bary=False,
                show_progress=False,
                default_init=False)
    
    simu.x += F*dt + F_inc*dt
    
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




