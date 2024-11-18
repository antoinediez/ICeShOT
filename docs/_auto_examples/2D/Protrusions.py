"""
Random protrusions
============================================

.. video:: ../../_static/SMV19_Protrusions.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

""" 

# sphinx_gallery_thumbnail_path = '_static/random_protrusions.png'

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
from iceshot.cells import DataPoints
from pykeops.torch import LazyTensor
from tqdm import tqdm 

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
    
# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB
    
simu_name = "simu_Protrusions"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N = 18
M = 300 

seeds = torch.rand((N,2))
source = sample.sample_grid(M)
vol_x = torch.ones(N)
vol_x *= 0.28/vol_x.sum()

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

K_circ = 6
dth = 2*math.pi/K_circ
th = torch.arange(-math.pi,math.pi,step=dth)
random_amplitude = 0.1*torch.rand((simu.N_crystals,K_circ)) 
lazy_th = LazyTensor(th[None,None,:])
lazy_random = LazyTensor(random_amplitude[:,None,:])

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


#=========================== RUN ===========================#

x0 = simu.x.detach().clone()
th0 = torch.atan2(simu.axis[:,1],simu.axis[:,0])
lazy_th0 = LazyTensor(th0[:,None,None])

pro = th0[:,None] + th[None,:]
x_pro = (simu.x[:,0].reshape((simu.N_cells,1)) + R0*torch.cos(pro[:simu.N_cells]))
y_pro = (simu.x[:,1].reshape((simu.N_cells,1)) + R0*torch.sin(pro[:simu.N_cells]))

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
    
    random_amplitude += 8.0 * dt * 2*(torch.rand((simu.N_crystals,K_circ)) - 0.5)
    random_amplitude[:] = torch.min(torch.max(torch.tensor([0.0]),random_amplitude),torch.tensor([5.0*R0]))
    XY = simu.lazy_XY()
    atanXY = (XY[:,:,1].atan2(XY[:,:,0]) - lazy_th0).mod(2*math.pi,-math.pi)
    bias_lazy = (-(((-(atanXY - lazy_th) ** 2 / 0.06).exp()/math.sqrt(math.pi*0.06)) * lazy_random).sum(-1)).exp()

    cost,grad_cost = solver.cost_matrix(simu)
    
    F_inc = solver.lloyd_step(simu,
            cost_matrix=(cost*bias_lazy,grad_cost),
            sinkhorn_algo=ot_algo,cap=None,
            tau=1.0/simu.R_mean,
            to_bary=False,
            show_progress=False,
            default_init=di)
    
    arange = torch.arange(0,simu.N_cells,1)
    pro = th0[:,None] + th[None,:]
    x_pro = (simu.x[:,0].reshape((simu.N_cells,1)) + R0*torch.cos(pro[:simu.N_cells]))
    y_pro = (simu.x[:,1].reshape((simu.N_cells,1)) + R0*torch.sin(pro[:simu.N_cells]))
    
    random_amplitude[:simu.N_cells][x_pro<0.01] = 0.0
    random_amplitude[:simu.N_cells][x_pro>0.99] = 0.0
    random_amplitude[:simu.N_cells][y_pro<0.01] = 0.0
    random_amplitude[:simu.N_cells][y_pro>0.99] = 0.0
    
    XY = simu.lazy_XY()
    am = simu.allocation_matrix()
    out = (XY ** 2).sum(-1) - (R0 ** 2)
    dist = (XY ** 2).sum(-1).sqrt() + 0.000001
    force = K_circ * (XY/dist * out.relu().sqrt() * am).sum(1) / (0.0000001 + (out.step() * am).sum(1)).reshape((simu.N_crystals,1))
    
    simu.x += Finc0*F_inc*dt + Fpro0*force*dt
    print(f"Maximal protrusion force: {torch.max(torch.norm(Fpro0*force,dim=1))}")
    print(f"Maximal incompressibility force: {torch.max(torch.norm(Finc0*F_inc,dim=1))}")
    print(f"Average force: {torch.norm(Finc0*F_inc + Fpro0*force,dim=1).mean()}")
    
    
    random_amplitude[:] -= amp_decay*dt*random_amplitude[:]
    
    simu.axis += math.sqrt(2*diff*dt)*torch.randn((N,2))
    simu.axis /= torch.norm(simu.axis,dim=1).reshape((N,1))
    
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

    
        
