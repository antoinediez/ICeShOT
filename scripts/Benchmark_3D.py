import os 
import sys
sys.path.append("..")
import time
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
import tifffile as tif
import pyvista as pv
from pyvista import themes

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
    
ot_algo = OT.LBFGSB

def benchmark(N,M,T=10,dt=0.005,plot_every=2,bsr=False):
    simu_name = f"simu_Benchmark_3D_{N}_{M}"
    os.mkdir(simu_name)
    os.mkdir(simu_name+"/frames")
    os.mkdir(simu_name+"/data")

    d = 3
    
    seeds = torch.rand(N,d)
    source = sample.sample_grid(M,dim=d,device=seeds.device)

    vol_x = 0.2 + 0.8*torch.rand(N)
    vol_x *= 0.8/vol_x.sum()

    simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",ar=torch.ones(N),
    bc=None
    )

    simu.orientation = simu.orientation_from_axis()

    min_ar = 1.0
    max_ar = 4.0
    min_ar = torch.tensor([min_ar],dtype=simu.x.dtype,device=simu.x.device)
    max_ar = torch.tensor([max_ar],dtype=simu.x.dtype,device=simu.x.device)

    p = 2
    eng = torch.linspace(3.0,4.0,N)

    
    cost_params = {
    "p" : p,
    "scaling" : "volume",
    "C" : eng
    }

    solver = OT_solver(
    n_sinkhorn=300,n_sinkhorn_last=2000,n_lloyds=5,s0=2.0,
    cost_function=costs.anisotropic_power_cost,cost_params=cost_params
    )

    t = 0.0
    t_iter = 0
    t_plot = 0

    v0 = 0.3

    data = {"pos" : [],
        "axis" : [],
        "ar" : []}
    
    #==================== Plot config ======================#
    pv.global_theme.volume_mapper = 'fixed_point'
    pv.global_theme.render_lines_as_tubes = True
    cmap0 = plt.cm.hsv
    
    off_screen = True
    plotter = pv.Plotter(lighting='three lights', off_screen=off_screen, image_scale=2)
    newcolors = np.zeros((N+1, 4))
    for n in range(N):
        # newcolors[n+1,:3] = 0.1 + 0.8*np.random.rand(3)
        newcolors[n,:] = np.array(cmap0(n/N))
        newcolors[n+1,3] = 1.0

    cmap = ListedColormap(newcolors)
    
    def plot_cells(p,img,**kwargs):
        img = np.pad(img,1,mode='constant',constant_values=-1.0)
        p.add_volume(img,shade=True,cmap=cmap,opacity='foreground',clim=(0,N-1),diffuse=0.85,**kwargs)

    box = pv.Cube(center=(M/2,M/2,M/2),x_length=M+2,y_length=M+2,z_length=M+2)

    #======================================================#

    solver.solve(simu,
                sinkhorn_algo=OT.LBFGSB,cap=None,
                tau=1.0,
                to_bary=True,
                show_progress=False,
                default_init=False,
                weight=1.0,
                bsr=True)
    
    simu.labels[simu.labels==torch.max(simu.labels)] = -1.0
    plot_cells(plotter,simu.labels.reshape(M,M,M).cpu().numpy())
    plotter.add_mesh(box, color='k', style='wireframe', line_width=5)
    plotter.remove_scalar_bar()
    plotter.screenshot(simu_name + f'/frames/t_{t_plot}.png')
    plotter.clear_actors()

    t += dt
    t_iter += 1
    t_plot += 1

    solver.n_lloyds = 1
    
    #======================================================#

    while t<T:
        print("--------------------------",flush=True)
        print(f"t={t}",flush=True)
        print("--------------------------",flush=True)

        plotting_time = t_iter%plot_every==0

        if plotting_time:
            print("I plot.",flush=True)
            solver.n_sinkhorn_last = 250
            solver.n_sinkhorn = 250
            solver.s0 = 2.0
        else:
            print("I do not plot.",flush=True)
            solver.n_sinkhorn_last = 250
            solver.n_sinkhorn = 250
            solver.s0 = 2*simu.R_mean
            
        R = (simu.volumes[:-1]/(4./3.*math.pi)) ** (1./3.)

        stime = time.time()
        F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=None,
                tau=42.0/(R**2),
                to_bary=False,
                show_progress=False,
                default_init=False,
                stopping_criterion="average",
                tol=0.01,
                bsr=bsr)
        print(f"Solving incompressibility: {time.time()-stime} seconds",flush=True)
        print(f"Mean incompressiblity force: {torch.norm(F_inc,dim=1).mean().item()}",flush=True)
        
        simu.x += v0*simu.axis*dt + F_inc*dt

        stime = time.time()
        cov = simu.covariance_matrix(bsr=bsr)
        print(f"Computing covariance matrix: {time.time()-stime} seconds",flush=True)

        stime = time.time()
        cov /= (torch.det(cov) ** (1./3.)).reshape((simu.N_cells,1,1))
        L,Q = torch.linalg.eigh(cov)
        ar = (L[:,2]/torch.sqrt(L[:,0]*L[:,1])) ** (2./3.)
        axis = Q[:,:,-1]
        axis = (axis * simu.axis).sum(1).sign().reshape((simu.N_cells,1)) * axis

        simu.axis = axis
        simu.ar = ar

        simu.ar = torch.max(min_ar,torch.min(max_ar,simu.ar))
        simu.orientation = simu.orientation_from_axis()

        simu.labels[simu.labels==torch.max(simu.labels)] = -1.0
        print(f"Update parameters: {time.time()-stime} seconds",flush=True)
        
        if plotting_time:
            plot_cells(plotter,simu.labels.reshape(M,M,M).cpu().numpy())
            plotter.add_mesh(box, color='black', style='wireframe', line_width=5)
            plotter.remove_scalar_bar()
            plotter.screenshot(simu_name + f'/frames/t_{t_plot}.png')
            plotter.clear_actors()
            data = {"pos" : simu.x,
                    "axis" : simu.axis,
                    "ar" : simu.ar}
            with open(simu_name + f"/data/data_{t_plot}.pkl",'wb') as file:
                pickle.dump(data,file)
            # tif.imsave(simu_name + "/frames/"+f"t_{t_plot}.tif", simu.labels.reshape(M,M,M).cpu().numpy(), bigtiff=True)
            t_plot += 1
        t += dt
        t_iter += 1

    with open(simu_name + "/data/data_final.pkl",'wb') as file:
        pickle.dump(simu,file)

    # utils.make_video(simu_name=simu_name,video_name=simu_name)



start_time = time.time()

benchmark(N=101,M=128,T=10,dt=0.005,plot_every=2,bsr=True)

print(f"--------------",flush=True)
print(f"Total computation time: {time.time() - start_time} seconds.",flush=True)