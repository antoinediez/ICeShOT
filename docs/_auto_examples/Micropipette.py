"""
Micropipette experiment
============================================


.. video:: ../_static/SMV10_Micropipette.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

""" 

# sphinx_gallery_thumbnail_path = '_static/Micropipette_eq.png'


import os 
import sys
sys.path.append("..")
import math
import pickle
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

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB
    
simu_name = "simu_Micropipette"
os.mkdir(simu_name)

radius = 0.08
vol0 = math.pi*radius**2
    
h_tube = radius/2.0
l_tube = vol0/h_tube
x0 = 1 - l_tube

def crop_function(x):
    return (torch.abs(x[:,0] - (1-0.5*l_tube))<=0.5*l_tube).float()*(torch.abs(x[:,1] - 0.5)<=0.5*h_tube).float() + (torch.abs(x[:,1] - 0.5)<=3.0*radius).float()*(torch.abs(x[:,0] - (1 - l_tube - 1.05*radius))<=1.05*radius).float()

scale = l_tube*h_tube + 2*1.05*radius*2*3.0*radius

N_cells = 1
M_grid = 800
vol_grid_true = 1.0/(M_grid**2)

dim = 2
source = sample.sample_cropped_domain(crop_function,M_grid)
seeds = torch.tensor([
        [1-l_tube-radius,0.5],
        ])

vol = vol0/scale
vol_x = torch.tensor([vol])

p_all = [0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0]
v_all = [0.5]

data = []

cmap = utils.cmap_from_list(100,color_names=["k"])


for iv0 in range(len(v_all)):
    v0 = v_all[iv0]
    os.mkdir(simu_name + f"/v0_{v0}")
    
    T = l_tube/v0
    fig_graph, ax_graph = plt.subplots(figsize=(8,8))
    ax_graph.set_xlim(0,1.0)
    ax_graph.set_ylim(0,1.0)
    
    for ip in range(len(p_all)):
        p = p_all[ip]
        dir_name = simu_name + f"/v0_{v0}" + f"/p_{p}"
        os.mkdir(dir_name)
        os.mkdir(dir_name + "/frames")
        
        print("===================================================")
        print(f"p={p}")
        print(f"v0={v0}")
        print("===================================================")
        
        simu = cells.Cells(
            seeds=seeds,source=source,
            vol_x=vol_x,extra_space="void"
        )
        
        print(vol_grid_true/simu.vol_grid)
        
        cost_params = {
            "p" : p,
            "scaling" : "volume",
            "R" : radius,
            "C" : 1.0
        }
        
        solver = OT_solver(
            n_sinkhorn=300,n_sinkhorn_last=3000,n_lloyds=10,
            cost_function=costs.power_cost,cost_params=cost_params
        )

        simu.axis[0,:] = torch.tensor([1.0,0.0])

        t_all = []
        x_all = []
        t = 0.0
        t_iter = 0
        t_plot = 0
        dt = 0.005

        solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=None,
             tau=0.0,
             to_bary=True,
             show_progress=False)

        t_all.append(0.0)
        x_all.append((torch.max(simu.y[simu.labels==0,0]).item()-x0)/l_tube)
        data.append({"t" : t_all,
                     "x" : x_all,
                     "p" : p,
                     "v0" : v0}
                    )
        pickle.dump(data,open(simu_name+"/data.p","wb"))
        graph, = ax_graph.plot(t_all,x_all,'*')
        fig_graph.savefig(simu_name + f"/v0_{v0}" + "/graph.png")

        simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
            plot_pixels=True,plot_scat=True,plot_quiv=True,plot_boundary=False,
            scat_size=15,scat_color='k',
            r=None,K=5,boundary_color='k',
            plot_type="scatter",void_color=plt.cm.bone(0.75),M_grid=M_grid)

        simu_plot.fig.savefig(dir_name + "/frames/" + f"t_{t_plot}.png")

        t += dt
        t_iter += 1
        t_plot += 1

        while t<=T: 
            print("--------",flush=True)
            print(f"t={t}",flush=True)
            
            solver.n_sinkhorn_last = 2000
            solver.n_sinkhorn = 2000
            solver.s0 = 2.0
            
            print(solver.cost)
            print(solver.cost_params)
            
            F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=None,
                tau=0.3/radius * vol_grid_true/simu.vol_grid,
                to_bary=False,
                show_progress=False,
                default_init=False)
    
            simu.x += v0*simu.axis*dt + F_inc*dt
            
            print(f"Maximal incompressibility force: {torch.max(torch.norm(F_inc,dim=1))}")
            
            t_all.append(t/T)
            x_all.append((torch.max(simu.y[simu.labels==0,0]).item()-x0)/l_tube)
            data[-1] = {"t" : t_all,
                        "x" : x_all,
                        "p" : p,
                        "v0" : v0}
            pickle.dump(data,open(simu_name+"/data.p","wb"))
            graph.set_xdata(t_all)
            graph.set_ydata(x_all)
            fig_graph.savefig(simu_name + f"/v0_{v0}" + "/graph.png")
            
            simu_plot.update_plot(simu)
            simu_plot.fig.savefig(dir_name + "/frames/" + f"t_{t_plot}.png")
            t_plot += 1
            
            # if (len(x_all)>101):
            #     if (abs((x_all[-1] - x_all[-100])) < 0.001):
            #         break
            t += dt
            t_iter += 1
            
            print("--------\n",flush=True)
            
        utils.make_video(simu_name=dir_name,video_name="v0_" + str(v0) + "_p_" + str(p))