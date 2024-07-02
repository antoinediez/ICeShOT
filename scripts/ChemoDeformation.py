"""
Chemotaxis induced by deformations
============================================

We consider a chemo-attractant density :math:`u(x)` and we assume that a particle located at :math:`x` can sense the local gradient along the directions :math:`{x}-{x}_i` (defined through a finite difference formula). 
The only force is the incompresisbility force. We introduce the biased cost potential :math:`\varphi = -\\log c`

.. math::

    \\varphi({x},{x}_i) = \\varphi_0({x},{x}_i) + \\beta f\\left(\\frac{u({x} - u({x}_i)}{x - x_i}\\right),
    
with the base potential :math:`\\varphi_0(x,y) = -2\\log |x - y|`, a constant :math:`\\beta>0` and a function :math:`f` which models how the gradient affects the deformation.

With :math:`f(\\delta)= \\max(0,\\delta)`, particles move with an elongated shape. 

.. video:: ../_static/SMV17_ChemoDeformation_long.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

With :math:`f(\\delta)=  \\max(0,-\\delta)^2 + \\max(0,\\delta)`, particles move with a fan-like shape. 


.. video:: ../_static/SMV18_ChemoDeformation_fan.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

**Related reference**

L. Yang, J. C. Effler, B. L. Kutscher, S. E. Sullivan, D. N. Robinson, and P. A. Iglesias. “Modeling Cellular Deformations Using the Level Set Formalism”. BMC Syst. Biol. 2.1 (2008)

""" 

# sphinx_gallery_thumbnail_path = '_static/ChemoDeformation_long.png'

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

simu_name = "simu_ChemoDeformation"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 10
M = 512 

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
vol_x = torch.ones(N)
vol_x *= 0.1/vol_x.sum()

R0 = math.sqrt(vol_x[-1].item()/math.pi)

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc=None
)

p = 2
cost_params = {
    "p" : p,
    "scaling" : "volume",
    "R" : math.sqrt(simu.volumes[0].item()/math.pi)
}

solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=1000,n_lloyds=10,s0=2.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)

def density(x,s=0.2,d=2):
    return 1/(2*math.pi*(s**2))**(d/2) * torch.exp(-((x-0.5)**2).sum(1)/(2*s**2))

x = y = torch.linspace(0.5/M,1-0.5/M,M)
Z = torch.zeros((M,M))
for i in range(M):
    for j in range(M):
        Z[i,j] = density(torch.tensor([[x[i],y[j]]]))


def lazy_grad(dx,dy,XY):
    lazy_dx = LazyTensor(dx[:,None,None])
    lazy_dy = LazyTensor(dy[None,:,None])
    return (lazy_dy - lazy_dx)/(XY ** 2).sum(-1).sqrt()


cmap = utils.cmap_from_list(N,color_names=["tab:blue"])

T = 10.0
dt = 0.01
plot_every = 1
t = 0.0
t_iter = 0
t_plot = 0

Finc0 = 0.6
Fpro0 = 1.6
amp_decay = 1.2
diff = 14.0
#======================= INITIALISE ========================#

tau0 = 1.0
solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=None,
             tau=tau0,
             to_bary=True,
             show_progress=False)

simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
                 plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=True,
                 scat_size=15,scat_color='k',
                 r=None,K=5,boundary_color='k',
                 plot_type="imshow",void_color='w')

chemo = simu_plot.ax.imshow(Z.cpu().numpy().transpose(),origin='lower', cmap=plt.cm.magma,alpha=0.6)

#====================== RUN =================================#


while t<T: 
    print("--------",flush=True)
    print(f"t={t}",flush=True)
    
    
    plotting_time = t_iter%plot_every==0
    
    if plotting_time:
        print("I plot.",flush=True)
        solver.n_sinkhorn_last = 2000
        solver.n_sinkhorn = 2000
        solver.s0 = 1.5
        di = False
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 300
        solver.n_sinkhorn = 300
        solver.s0 = 2*simu.R_mean
        di = False
        
        
    XY = simu.lazy_XY()
    dx = density(simu.x)
    dy = density(simu.y)
    grad = lazy_grad(dx,dy,XY)
    # bias_lazy = (-0.02*grad.relu()**2).exp()    # elongated shape
    # bias_lazy = (-0.2*grad).exp()
    bias_lazy = (0.02*((-grad).relu()**2 + (grad.relu()))).exp()    # fan-like shape

    cost,grad_cost = solver.cost_matrix(simu)
        
    F_inc = solver.lloyd_step(simu,
            cost_matrix=(cost*bias_lazy,grad_cost),
            sinkhorn_algo=ot_algo,cap=None,
            tau=1/simu.R_mean,
            to_bary=True,weight=1.0,
            show_progress=False,
            default_init=False)
    
    simu.x += F_inc * dt
    
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