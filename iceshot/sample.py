import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sample_uniform(N,d,dtype=torch.float32,device=DEVICE):
    return torch.rand((N,d),dtype=dtype,device=device)

def sample_grid(n, dim=2,dtype=torch.float32,device=DEVICE):
    grid_points = torch.linspace(0.5/n,1-0.5/n,n,dtype=dtype,device=device)
    grid_points = torch.stack(
        torch.meshgrid((grid_points,) * dim, indexing="ij"), dim=-1
    )
    grid_points = grid_points.reshape(-1, dim)
    return grid_points

def sample_Aorientations_mat(N,r=3.0,dtype=torch.float32,device=DEVICE):
    diag = torch.diag(torch.tensor([1.0/r,r]),dtype=dtype,device=device)
    rot = 2*np.pi*torch.rand(N,dtype=dtype,device=device)
    cos = torch.cos(rot).reshape((N,1))
    sin = torch.sin(rot).reshape((N,1))
    e1 = torch.cat((cos,sin),axis=1)
    e2 = torch.cat((-sin,cos),axis=1)
    R = torch.cat((e1.reshape((N,2,1)),e2.reshape((N,2,1))),axis=2)
    return R @ diag @ torch.transpose(R,1,2)

def sample_cropped_domain(crop_function,n,dim=2,dtype=torch.float32,device=DEVICE):
    grid_points = sample_grid(n,dim=dim,dtype=dtype,device=device)
    return grid_points[crop_function(grid_points)>0,:]

def sample_cropped_crystals(crop_function,N,dim=2,dtype=torch.float32,device=DEVICE):
    x = torch.rand(N,dim,dtype=dtype,device=device)
    reject = crop_function(x)<=0
    while reject.sum()>0:
        rej = reject.clone()
        x[rej,:] = torch.rand(rej.sum(),dim,dtype=dtype,device=device)
        reject[rej] = crop_function(x[rej,:])<0
    return x