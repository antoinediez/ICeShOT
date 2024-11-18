"""
Engulfment due to surface tension effects
============================================

We consider a two-population system with surface interactions. The main force models surface tension-like effects by computing a pressure-like force acting on the centroid of each cell and computed as the sum of elementary forces orthognal to its boundary.
The different relative magnitudes of the forces produced at the interfaces between different populations or between cells and the medium classicaly leads to various sorting effects. 

.. video:: ../../_static/SMV23_Engulfment.mp4
    :autoplay:
    :loop:
    :muted:
    :width: 400
    
|

**Related references**

G. W. Brodland. “The Differential Interfacial Tension Hypothesis (DITH): A Comprehensive Theory for the Self-Rearrangement of Embryonic Cells and Tissues”. Journal of Biomechanical Engineering 124.2 (2002)


R. Z. Mohammad, H. Murakawa, K. Svadlenka, and H. Togashi. “A Numerical Algorithm for Modeling Cellular Rearrangements in Tissue Mor- phogenesis”. Commun. Biol. 5.1 (2022)

F. Graner and J. A. Glazier. “Simulation of Biological Cell Sorting Using a Two-Dimensional Extended Potts Model”. Phys. Rev. Lett. 69.13 (1992)

D. Sulsky, S. Childress, and J. Percus. “A Model of Cell Sorting”. J. Theoret. Biol. 106.3 (1984)
""" 

# sphinx_gallery_thumbnail_path = '_static/Engulfment_t500.png'


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
from pykeops.torch import LazyTensor


use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"

# ot_algo = OT.sinkhorn_zerolast
ot_algo = OT.LBFGSB

simu_name = "simu_Engulfment"
os.mkdir(simu_name)
os.mkdir(simu_name+"/frames")
os.mkdir(simu_name+"/data")

N1 = 14
N2 = 154
N3 = 0
N = N1+N2+N3
M = 512

# cmap = utils.cmap_from_list(N1,N2,N3,color_names=["tab:orange","tab:blue","tab:gray"])
cmap = utils.cmap_from_list(N1,N2,color_names=["tab:orange","tab:blue"])

source = sample.sample_grid(M)

vol_x = torch.ones(N)
vol_x *= 0.34/vol_x.sum()
R0 = math.sqrt(vol_x[-1].item()/math.pi)
eps_ifc=None
eps_ifc = R0/4.0
print(eps_ifc * M)
if eps_ifc * M < 3: 
    raise ValueError()

seeds = torch.rand(N,2)
r10 = (vol_x[:N1].sum()/math.pi)
r1 = torch.rand((N1,1))*r10*0.5
r20 = (vol_x[N1:(N1+N2)].sum()/math.pi)
r2 = torch.rand((N2,1))* r20*0.5
ag1 = torch.rand((N1,1))*2*math.pi
ag2 = torch.rand((N2,1))*2*math.pi
r12 = (torch.sqrt(r10) + torch.sqrt(r20))/2.0
seeds[:N1,:] = torch.sqrt(r1)*torch.cat((torch.cos(ag1),torch.sin(ag1)),dim=1)
seeds[:N1,:] += torch.tensor([[1.1*torch.sqrt(r10),0.5]])
seeds[N1:(N1+N2),:] = torch.sqrt(r2)*torch.cat((torch.cos(ag2),torch.sin(ag2)),dim=1)
seeds[N1:(N1+N2),:] += torch.tensor([[1.0-1.2*torch.sqrt(r20),0.5]])

simu = cells.Cells(
    seeds=seeds,source=source,
    vol_x=vol_x,extra_space="void",
    bc=None
)

p=2
cost_params = {
    "scaling" : "volume",
    "R" : math.sqrt(simu.volumes[0].item()/math.pi),
}
solver = OT_solver(
    n_sinkhorn=1000,n_sinkhorn_last=3000,n_lloyds=7,s0=2.0,
    cost_function=costs.l2_cost,cost_params=cost_params
)


T = 10.0
dt = 0.0002
plot_every = 5
t = 0.0
t_iter = 0
t_plot = 0
cap = None

Finc0 = 0.5
F0_jct = 1.0 / 7.0
F0_ifc = 1.0 / 7.0

g11 = 1.0
g22 = 1.0
g12 = 5.0
g10 = 30.0
g20 = 7.0

print(f"g11={g11}")
print(f"g10={g10}")
print(f"g22={g22}")
print(f"g20={g20}")
print(f"g12={g12}")

if g12<g11/2 or g12<g22/2:
    raise ValueError("It cannot sort")
if g12>(g10 - g20) or g12>g10:
    raise ValueError("Stupid")


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
tau0[:(N1+N2)] = 0.14
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

simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")

with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
    pickle.dump(simu,file)
    
t += dt
t_iter += 1
t_plot += 1

solver.n_lloyds = 1

with open(simu_name + f"/params.pkl",'wb') as file:
    pickle.dump(solver,file)

#=========================== RUN ===========================#

while t<T:
    print("--------------------------",flush=True)
    print(f"t={t}",flush=True)
    print("--------------------------",flush=True)
    
    plotting_time = t_iter%plot_every==0
    
    if t_plot==10:
        x1max = torch.max(simu.x[:N1,0]).item()
        x2min = torch.min(simu.x[N1:(N1+N2),0]).item()
        simu.x[:N1,0] += 0.94*((x2min - x1max) - 2*simu.R_mean)
        plotting_time = True
        # F0 = 0.2
    
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
        solver.s0 = simu.R_mean
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

    simu.x += F_interface*dt + F_jct*dt + Finc0*F_inc*dt
    print(f"Maximal interface force: {torch.max(torch.norm(F_interface,dim=1))}")
    print(f"Maximal junction force: {torch.max(torch.norm(F_jct,dim=1))}")
    print(f"Maximal incompressibility force: {torch.max(torch.norm(Finc0*F_inc,dim=1))}")
    print(f"Average force: {torch.norm(Finc0*F_inc + F_jct + F_interface,dim=1).mean()}")

    if plotting_time:
        simu_plot.update_plot(simu)
        simu_plot.fig.savefig(simu_name + "/frames/" + f"t_{t_plot}.png")
        with open(simu_name + "/data/" + f"data_{t_plot}.pkl",'wb') as file:
            pickle.dump(simu,file)
        t_plot += 1

    t += dt
    t_iter += 1
    
utils.make_video(simu_name=simu_name,video_name=simu_name)