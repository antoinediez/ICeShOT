"""
Active Brownian Particles
============================================

We consider the motion of :math:`N` **active deformable spheres** in a periodic box with different deformability properties.
The particle are defined by their positions :math:`x_i` and Brownian active directions of motion :math:`n_i`, which follow the following set of stochastic differential equations:

.. math::

    \\mathrm{d}{x}_i = c_0 {n}_i\\mathrm{d} t - \\tau\\nabla_{{x}_i}\mathcal{T}_c(\\hat{\mu})\\mathrm{d} t

.. math::

    \\mathrm{d}{n}_i = (\\mathrm{Id} - {n}_i{n}_i^\\mathrm{T})\\circ \\sqrt{2\\sigma}\\mathrm{d} B^i_t,

The incompressibility force :math:`\\nabla_{{x}_i}\mathcal{T}_c(\\hat{\mu})` is associated to the optimal transport cost

.. math:: 

    c(x,y) = |y-x|^p
    
where the coefficient :math:`p` sets the deformability of the particles. Increasing :math:`p` leads to a transition from a liquid-like state to a crystal-like state.


With :math:`p=0.5`, particles are easy to deform.

.. video:: ../../_static/SMV11_ActiveBrownian_p05.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`p=2`, 

.. video:: ../../_static/SMV12_ActiveBrownian_p2.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`p=8`, particles behave as hard-spheres. 

.. video:: ../../_static/SMV13_ActiveBrownian_p8.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

**Related references:**

N. Saito and S. Ishihara. “Active Deformable Cells Undergo Cell Shape Transition Associated with Percolation of Topological Defects”, Science Advances 10.19 (2024)

D. Bi, X. Yang, M. C. Marchetti, and M. L. Manning. “Motility-Driven Glass and Jamming Transitions in Biological Tissues”. Physical Review X 6.2 (2016)
""" 

# sphinx_gallery_thumbnail_path = '_static/ActiveBrownian_p8.png'

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

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB

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
        solver.n_sinkhorn_last = 2000
        solver.n_sinkhorn = 2000
        solver.s0 = 2.0
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 250
        solver.n_sinkhorn = 250
        solver.s0 = 2*simu.R_mean
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=cap,
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



