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
    
simu_name = "simu_SurfaceTension"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N1 = 123
N2 = 123
N = N1+N2

M = 600 

seeds = torch.rand((N,2))
r2 = torch.rand((N1+N2,1)) * 0.12
th = 2*math.pi*torch.rand((N1+N2,1))
seeds[:(N1+N2),:] = 0.5 + torch.sqrt(r2) * torch.cat((torch.cos(th),torch.sin(th)),dim=1)
source = sample.sample_grid(M)
vol_x = torch.ones(N)
vol_x *= 0.4/vol_x.sum()

R0 = math.sqrt(vol_x[-1].item()/math.pi)
# eps_ifc=None
eps_ifc = R0/4.0
print(eps_ifc * M)
if eps_ifc * M < 3: 
    raise ValueError()

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

T = 10.0
dt = 0.0005
plot_every = 5
t = 0.0
t_iter = 0
t_plot = 0
cap = None

Finc0 = 0.5
F0_jct = 1.0 / 7.0
F0_ifc = 1.0 / 7.0

# Checkerboard
# g11 = 7.0
# g22 = 7.0
# g12 = 1.0
# g10 = 30.0
# g20 = 30.0

## Sorting engulfment
g11 = 1.0
g22 = 1.0
g12 = 5.0
g10 = 30.0
g20 = 7.0


## Separation
# g11 = 1.0
# g22 = 1.0
# g12 = 12.0
# g10 = 5.0
# g20 = 5.0

# diff = 3.0*R0**2
diff = 0.0

cmap = utils.cmap_from_list(N1,N2,color_names=["tab:orange","tab:blue"])

#====================== FORCES =============================#

def compute_tension(i,j,N1,N2,g11,g22,g12,g10,g20):
    is1_i = float(i<N1)
    is2_i = float((i>=N1)&(i<N1+N2))
    is0_i = float(i>=N1+N2)
    
    is1_j = float(j<N1)
    is2_j = float((j>=N1)&(j<N1+N2))
    is0_j = float(j>=N1+N2)
    
    tij = g11*(is1_i*is1_j)\
        + g22*(is2_i*is2_i)\
        + g12*(is1_i*is2_j + is2_i*is1_j)\
        + g10*(is1_i*is0_j + is0_i*is1_j)\
        + g20*(is2_i*is0_j + is0_i*is2_j)\
            
    return tij

def compute_interface_force_ij(cells,ind_y,i,j,eps=None,p=2):
    if eps is None:
        eps = 5.0 * cells.vol_grid ** (1/cells.d)
    YY = LazyTensor(simu.y[ind_y,None,:]) - LazyTensor(simu.y[None,:,:])
    K = (-(YY**2).sum(-1) + eps**2).step()
    is_bound = K.sum(0).squeeze()>0.01
    yi = simu.y[is_bound,:] - simu.x[i,:] if i<simu.N_cells else torch.zeros_like(simu.y[is_bound,:])
    yj = simu.y[is_bound,:] - simu.x[j,:] if j<simu.N_cells else torch.zeros_like(simu.y[is_bound,:])
    gi = p * yi*(torch.norm(yi,dim=1).reshape((len(yi),1)) + 1e-8)**(p-2)
    gj = p * yj*(torch.norm(yj,dim=1).reshape((len(yj),1)) + 1e-8)**(p-2)
    normal = (gi - gj)/(torch.norm(gi-gj,dim=1).reshape((len(gi),1))+1e-8)
    zij = torch.norm(yi - yj,dim=1).reshape((len(yi),1))
    return -(normal/(zij+1e-8)).sum(0) * cells.vol_grid/eps
    
def compute_interface_force(cells,eps=None,p=2,g11=1.0,g22=1.0,g12=1.0,g10=1.0,g20=1.0,N1=1,N2=1):
    ind_y, ind_x = cells.extract_boundary()
    pairs, index = torch.unique(ind_x,dim=0,return_inverse=True)
    F = torch.zeros_like(cells.x)
    for k in range(len(pairs)):
        i = pairs[k,0].item()
        j = pairs[k,1].item()
        force_ij = compute_interface_force_ij(cells,ind_y[index==k],i,j,eps=eps,p=p)
        gij = compute_tension(i,j,N1,N2,g11,g22,g12,g10,g20)
        if i<cells.N_cells:
            F[i,:] += force_ij * gij
        if j<cells.N_cells:
            F[j,:] -= force_ij * gij
    return F


def compute_orthogonal(x):
    x_p = torch.zeros_like(x)
    x_p[:,0] = -x[:,1]
    x_p[:,1] = x[:,0]
    return x_p

def compute_node(yx1,yx2,yx3,yx1b=None,yx2b=None,yx3b=None):
    if yx1b is None:
        yx1b = yx1
    if yx2b is None:
        yx2b = yx2
    if yx3b is None:
        yx3b = yx3
    x12 = normalize(yx2 - yx1)
    x23 = normalize(yx3 - yx2)
    x31 = normalize(yx1 - yx3)
    
    x12_p = compute_orthogonal(x12)
    x23_p = compute_orthogonal(x23)
    x31_p = compute_orthogonal(x31)

    # C1 = (-yx2*x12_p).sum(1)>=(-yx2*x23_p).sum(1)*(x12_p*x23_p).sum(1)
    # C2 = (-yx2*x23_p).sum(1)>=(-yx2*x12_p).sum(1)*(x12_p*x23_p).sum(1)
    C1 = (-yx2b*x12_p).sum(1)>=(-yx2b*x23_p).sum(1)*(x12_p*x23_p).sum(1)
    C2 = (-yx2b*x23_p).sum(1)>=(-yx2b*x12_p).sum(1)*(x12_p*x23_p).sum(1)
    sgn = (2*(C1.float()*C2.float())-1.0).reshape((len(yx1),1))
    
    x12_p *= sgn
    x23_p *= sgn
    x31_p *= sgn
    
    return x12,x23,x31,x12_p,x23_p,x31_p

def compute_tension_ind(ind_x,i,j,N1,N2,g11,g22,g12,g10,g20):
    is1 = ind_x<N1
    is2 = (ind_x>=N1)&(ind_x<(N1+N2))
    is0 = ind_x>=N1+N2
    
    tij = g11*(is1[:,i].float()*is1[:,j].float())\
        + g22*(is2[:,i].float()*is2[:,j].float())\
        + g12*(is1[:,i].float()*is2[:,j].float() + is2[:,i].float()*is1[:,j].float())\
        + g10*(is1[:,i].float()*is0[:,j].float() + is0[:,i].float()*is1[:,j].float())\
        + g20*(is2[:,i].float()*is0[:,j].float() + is0[:,i].float()*is2[:,j].float())\
            
    return tij.reshape((len(ind_x),1))

def normalize(x):
    y = torch.zeros_like(x)
    norm = torch.norm(x,dim=1)
    y[norm>0,:] = x[norm>0,:]/norm[norm>0].reshape((len(x[norm>0,:]),1))
    return y
    
def triple_junction_force(cells,y_junction,ind_x,N1,N2,g11,g22,g12,g10,g20,p=2):
    
    yx1 = torch.zeros_like(y_junction)
    yx1[ind_x[:,0]<N,:] = y_junction[ind_x[:,0]<N,:] - cells.x[ind_x[ind_x[:,0]<N,0],:]
    # yx1 = y_junction - cells.x[ind_x[:,0],:]
    yx2 = y_junction - cells.x[ind_x[:,1],:]
    yx3 = y_junction - cells.x[ind_x[:,2],:]
    
    yx1b = torch.zeros_like(y_junction)
    barycenters = cells.barycenters()
    yx1b[ind_x[:,0]<N,:] = y_junction[ind_x[:,0]<N,:] - barycenters[ind_x[ind_x[:,0]<N,0],:]
    # yx1 = y_junction - cells.x[ind_x[:,0],:]
    yx2b = y_junction - barycenters[ind_x[:,1],:]
    yx3b = y_junction - barycenters[ind_x[:,2],:]
    
    x12,x23,x31,x12_p,x23_p,x31_p = compute_node(yx1,yx2,yx3,yx1b=yx1b,yx2b=yx2b,yx3b=yx3b)
    
    if p!=2:
        yx1 = p*(torch.norm(yx1,dim=1).reshape((len(y_junction),1)) + 1e-8)**(p-2)*yx1
        yx2 = p*(torch.norm(yx2,dim=1).reshape((len(y_junction),1)) + 1e-8)**(p-2)*yx2
        yx3 = p*(torch.norm(yx3,dim=1).reshape((len(y_junction),1)) + 1e-8)**(p-2)*yx3

    
    f12 = compute_tension_ind(ind_x,0,1,N1,N2,g11,g22,g12,g10,g20)*x12_p
    f23 = compute_tension_ind(ind_x,1,2,N1,N2,g11,g22,g12,g10,g20)*x23_p
    f31 = compute_tension_ind(ind_x,2,0,N1,N2,g11,g22,g12,g10,g20)*x31_p
    
    force = f12 + f23 + f31
    
    # x12,x23,x31,x12_p,x23_p,x31_p = compute_node(yx1b,yx2b,yx3b)
    # yx1, yx2, yx3 = yx1b, yx2b, yx3b
    
    yx1_n = normalize(yx1)
    yx2_n = normalize(yx2)
    yx3_n = normalize(yx3)
    
    # f1b = (force * yx1_n).sum(1).reshape(len(force),1) * yx1_n
    # f2b = (force * yx2_n).sum(1).reshape(len(force),1) * yx2_n
    # f3b = (force * yx3_n).sum(1).reshape(len(force),1) * yx3_n
    f1 = (force * yx1_n).sum(1).reshape(len(force),1) * yx1_n
    f2 = (force * yx2_n).sum(1).reshape(len(force),1) * yx2_n
    f3 = (force * yx3_n).sum(1).reshape(len(force),1) * yx3_n
    
    return f1,f2,f3

def compute_triplejunction_force(cells,y_junction,ind_x,N1=1,N2=1,g11=1.0,g22=1.0,g12=1.0,g10=1.0,g20=1.0,p=2):
    f1,f2,f3 = triple_junction_force(cells,y_junction,ind_x,N1,N2,g11,g22,g12,g10,g20,p=p)
    F = torch.zeros_like(cells.x)
    for i in range(len(y_junction)):
        if ind_x[i,0]<N1+N2:
            F[ind_x[i,0],:] += f1[i,:]
        F[ind_x[i,1],:] += f2[i,:]
        F[ind_x[i,2],:] += f3[i,:]
    return F


#======================= INITIALISE ========================#

tau0 = torch.ones(N)
tau0[:(N1+N2)] = 1.0
# tau0[:(N1+N2)] = 0.5
# tau0[:(N1+N2)] = 0.25
solver.solve(simu,
             sinkhorn_algo=ot_algo,cap=cap,
             tau=tau0,
             to_bary=True,
             show_progress=False)

simu_plot = plot_cells.CellPlot(simu,figsize=8,cmap=cmap,
                 plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=True,
                 scat_size=15,scat_color='k',
                 r=None,K=5,boundary_color='k',
                 plot_type="imshow",void_color='w')


# F_interface = F0_ifc*compute_interface_force(simu,g11=g11,g22=g22,g12=g12,g10=g10,g20=g20,N1=N1,N2=N2,p=p,eps=eps_ifc)
# f_int_pl = simu_plot.ax.quiver(M*simu.x[:,0].cpu(),M*simu.x[:,1].cpu(),F_interface[:,0].cpu(),F_interface[:,1].cpu(),color='c',scale=42.0,scale_units='width')

# junctions,ind_x = simu.clusterized_triple_junctions(
#     r0=1.1*(1.0/simu.M_grid) ** (1.0/simu.d),
#     r=R0/3.0,
#     )

# jct = simu_plot.ax.scatter(M*junctions[:,0].cpu(),M*junctions[:,1].cpu(),c='r',s=10)
# F_jct = F0_jct*compute_triplejunction_force(simu,junctions,ind_x,N1=N1,N2=N2,g11=g11,g22=g22,g12=g12,g10=g10,g20=g20,p=p)
# f_jct_pl = simu_plot.ax.quiver(M*simu.x[:,0].cpu(),M*simu.x[:,1].cpu(),F_jct[:,0].cpu(),F_jct[:,1].cpu(),color='m',scale=42.0,scale_units='width')

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
        solver.s0 = 1.5
        di = False
    else:
        print("I do not plot.",flush=True)
        solver.n_sinkhorn_last = 300
        solver.n_sinkhorn = 300
        solver.s0 = 2*simu.R_mean
        di = False
    
    F_inc = solver.lloyd_step(simu,
                sinkhorn_algo=ot_algo,cap=cap,
                tau=1.0/simu.R_mean,
                to_bary=False,
                show_progress=False,
                default_init=di)
    
    junctions,ind_x = simu.clusterized_triple_junctions(
        r0=1.1*(1.0/simu.M_grid) ** (1.0/simu.d),
        r=R0/3.0,
    )
    
    F_interface = F0_ifc*compute_interface_force(simu,g11=g11,g22=g22,g12=g12,g10=g10,g20=g20,N1=N1,N2=N2,p=p,eps=eps_ifc)
    F_jct = F0_jct*compute_triplejunction_force(simu,junctions,ind_x,N1=N1,N2=N2,g11=g11,g22=g22,g12=g12,g10=g10,g20=g20,p=p)

    noise = math.sqrt(2*diff*dt) * torch.randn((N1+N2,2))
    
    simu.x += F_interface*dt + F_jct*dt + Finc0*F_inc*dt + noise
    print(f"Maximal interface force: {torch.max(torch.norm(F_interface,dim=1))}")
    print(f"Maximal junction force: {torch.max(torch.norm(F_jct,dim=1))}")
    print(f"Maximal incompressibility force: {torch.max(torch.norm(Finc0*F_inc,dim=1))}")
    print(f"Average force: {torch.norm(Finc0*F_inc + F_jct + F_interface,dim=1).mean()}")
    print(f"Noise: {torch.norm(noise/dt,dim=1).mean()}")
    if plotting_time:
        simu_plot.update_plot(simu)
        # jct.set_offsets(M*junctions.cpu())
        # f_int_pl.set_offsets(M*simu.x.cpu())
        # f_int_pl.set_UVC(F_interface[:,0].cpu(),F_interface[:,1].cpu())
        # f_jct_pl.set_offsets(M*simu.x.cpu())
        # f_jct_pl.set_UVC(F_jct[:,0].cpu(),F_jct[:,1].cpu())
            
        simu_plot.fig
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)

        t_plot += 1

    t += dt
    t_iter += 1
    
utils.make_video(simu_name=simu_name,video_name=simu_name)




