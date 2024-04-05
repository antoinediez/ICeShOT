import numpy as np 
import torch
from pykeops.torch import LazyTensor
from tqdm import tqdm
from . import utils

def LBFGSB(C_xy,scales,f_x,g_y,a=None,b=None,default_init=False,show_progress=False,max_epoch=10,tol=0.01,stopping_criterion="average",**kwargs):
    n_iter = len(scales)
    N, M = C_xy.shape
    assert f_x.shape == (N+1,), f"`f_x` must be of size {N+1}"
    assert g_y.shape == (M,), f"`g_y` must be of size {M}"
    
    if a is None:
        a = torch.ones(N+1,device=f_x.device,dtype=f_x.dtype)/(N+1)
    if b is None:
        b = torch.ones(M,device=f_x.device,dtype=f_x.dtype)/M
    
    assert a.shape == (N+1,), f"`a` must be of size {N+1}"
    assert b.shape == (M,), f"`b` must be of size {M}"

    def dual_cost(seed_potentials):
        # f_i : add a 0 for the medium to seed_potentials
        print(".", end="", flush=True)
        f_x = torch.cat(
            [seed_potentials, torch.zeros_like(seed_potentials[:1])]
        )
        mat = C_xy - f_x[:-1].view(N,1,1)
        # KeOps does not support backpropagation through the min reduction yet... ugly workaround...
        yi = mat.argmin(axis=0).reshape(M)
        lazy_x = LazyTensor(torch.arange(N,device=f_x.device,dtype=f_x.dtype)[:,None,None])
        lazy_labels = LazyTensor(yi[None,:,None].float())
        alloc = (-(lazy_x - lazy_labels).abs()).step()
        g_y = (mat * alloc).sum(0).reshape(M)
        g_y = torch.min(g_y,torch.zeros_like(g_y))
        res = (a * f_x).sum() + (b * g_y).sum()
        return res

    seed_potentials = torch.zeros_like(a[:-1]) if default_init else f_x[:-1]
    seed_potentials.requires_grad = True
    optimizer = torch.optim.LBFGS(
        [seed_potentials], max_iter=n_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = - dual_cost(seed_potentials)
        loss.backward()
        return loss
    
    for epoch in range(max_epoch):
        print(f"\nepoch = {epoch}",flush=True)
        optimizer.step(closure)
        
        # Compute the (lazy) allocation matrix
        mat = C_xy - f_x[:-1].view(N,1,1)
        m, am = mat.min_argmin(axis=0)
        m = m.squeeze()
        am = am.squeeze().float()
        labels = torch.zeros_like(am)
        labels[m<=-f_x[-1]] = am[m<=-f_x[-1]]
        labels[m>-f_x[-1]] = N + 42.0
        lazy_x = LazyTensor(torch.arange(N,device=f_x.device,dtype=f_x.dtype)[:,None,None])
        lazy_labels = LazyTensor(labels[None,:,None].float())
        alloc = (-(lazy_x - lazy_labels).abs()).step()

        # Compute the volume deviation 
        vols = alloc.sum(1).squeeze().reshape(N) * 1/M
        vol_dev = (vols - a[:-1]).abs()/a[:-1]
        vol_dev_max = vol_dev.max()
        vol_dev_mean = vol_dev.mean()
        
        if stopping_criterion == "average":
            stop = vol_dev_mean < tol
        elif stopping_criterion == "maximum":
            stop = vol_dev_max < tol
        else:
            raise ValueError("Stopping critertion must be `average` or `maximum`")
        if stop:
            print("\nSuccess!\n",flush=True)
            break
        
    if not stop:
        print("\nWarning: optimization did not converge...\n",flush=True)
        
    f_x = torch.cat([seed_potentials,torch.zeros_like(seed_potentials[:1])])
    mat = C_xy - f_x[:-1].view(N,1,1)
    yi = mat.argmin(axis=0).reshape(M)
    lazy_x = LazyTensor(torch.arange(N,device=f_x.device,dtype=f_x.dtype)[:,None,None])
    lazy_labels = LazyTensor(yi[None,:,None].float())
    alloc = (-(lazy_x - lazy_labels).abs()).step()
    g_y = (mat * alloc).sum(0).reshape(M)
    
    
def annealing_schedule(n_iter,M_grid,s0=4.0,device="cuda",dtype=torch.float32):
    # We start from s0 (> diameter of the unit cube) and decrease to the sampling length:
    log_scales = torch.linspace(np.log(s0),np.log(1/M_grid),n_iter,device=device,dtype=dtype)
    scales = log_scales.exp()
    # To be on the safe side, we add a few extra iterations at the end:
    scales = torch.cat((scales, 1 / M_grid * torch.ones(10,device=device,dtype=dtype)))
    return scales

    
def sinkhorn(C_xy,scales,f_x,g_y,a=None,b=None,default_init=True,show_progress=False,**kwargs):
    N, M = C_xy.shape
    assert f_x.shape == (N,), f"`f_x` must be of size {N}"
    assert g_y.shape == (M,), f"`g_y` must be of size {M}"
    
    if a is None:
        a = torch.ones(N,device=scales.device,dtype=scales.dtype)/N
    if b is None:
        b = torch.ones(M,device=scales.device,dtype=scales.dtype)/M
    
    assert a.shape == (N,), f"`a` must be of size {N}"
    assert b.shape == (M,), f"`b` must be of size {M}"
    
    if default_init:
        f_x[:] = C_xy @ b - 0.5*(a * (C_xy @ b)).sum()
        g_y[:] = C_xy.t() @ a - 0.5*(a * (C_xy @ b)).sum()
    
    # Log of the weights, (B, N) and (B, M):
    a_logs, b_logs = a.log(), b.log()

    # Encoding as symbolic tensors:
    # Dual potentials:
    f_i = LazyTensor(f_x.view(N, 1, 1))  # (N, 1, 1)
    g_j = LazyTensor(g_y.view(1, M, 1))  # (1, M, 1)
    # Log-weights:
    log_a_i = LazyTensor(a_logs.view(N, 1, 1))  # (N, 1, 1)
    log_b_j = LazyTensor(b_logs.view(1, M, 1))  # (1, M, 1)
    
    scales_prog = tqdm(scales) if show_progress else scales

    # Symmetric Sinkhorn iterations, written in the log-domain:
    for scale in scales_prog:
        eps = scale**2
        ft_x = -eps * ((g_j - C_xy) / eps + log_b_j).logsumexp(dim=1).squeeze(-1)
        gt_y = -eps * ((f_i - C_xy) / eps + log_a_i).logsumexp(dim=0).squeeze(-1)
        # Use in-place updates to keep a small memory footprint:
        f_x[:] = (f_x + ft_x) / 2
        g_y[:] = (g_y + gt_y) / 2
        
def sinkhorn_zerolast(C_xy,scales,f_x,g_y,a=None,b=None,default_init=True,show_progress=False,**kwargs):
    N, M = C_xy.shape
    assert f_x.shape == (N+1,), f"`f_x` must be of size {N+1}"
    assert g_y.shape == (M,), f"`g_y` must be of size {M}"
    
    if a is None:
        a = torch.ones(N+1,device=scales.device,dtype=scales.dtype)/(N+1)
    if b is None:
        b = torch.ones(M,device=scales.device,dtype=scales.dtype)/M
    
    assert a.shape == (N+1,), f"`a` must be of size {N+1}"
    assert b.shape == (M,), f"`b` must be of size {M}"
    
    if default_init:
        f_x[:-1] = C_xy @ b - 0.5*(a[:-1] * (C_xy @ b)).sum()
        f_x[-1] = - 0.5*(a[-1] * (C_xy @ b)).sum()
        g_y[:] = C_xy.t() @ a[:-1] - 0.5*(a[:-1] * (C_xy @ b)).sum()
    
    # Log of the weights, (B, N) and (B, M):
    a_logs, b_logs = a.log(), b.log()
    
    # Encoding as symbolic tensors:
    # Dual potentials:
    f_i = LazyTensor(f_x[:-1].view(N, 1, 1))  # (N, 1, 1)
    g_j = LazyTensor(g_y.view(1, M, 1))  # (1, M, 1)
    # Log-weights:
    log_a_i = LazyTensor(a_logs[:-1].view(N, 1, 1))  # (N+1, 1, 1)
    log_b_j = LazyTensor(b_logs.view(1, M, 1))  # (1, M, 1)
    
    scales_prog = tqdm(scales) if show_progress else scales

    # Symmetric Sinkhorn iterations, written in the log-domain:
    for scale in scales_prog:
        K = f_x[-1]
        f_x[:] = f_x - K
        g_y[:] = g_y + K
        eps = scale**2
        ft_x = -eps * ((g_j - C_xy) / eps + log_b_j).logsumexp(dim=1).squeeze(-1)
        LSE_g = ((f_i - C_xy) / eps + log_a_i).logsumexp(dim=0).squeeze(-1)
        # Use in-place updates to keep a small memory footprint:
        f_x[:-1] = (f_x[:-1] + ft_x) / 2
        f_x[-1] = (f_x[-1] + -eps * (g_y/ eps + b_logs).logsumexp(dim=0)) / 2
        f0 = a_logs[-1] + 1/eps*f_x[-1]
        g_y[:] = (g_y - eps*(utils.log1pexp(LSE_g - f0) + f0)) / 2
        
        
class OT_solver:
    
    def __init__(self,
        n_sinkhorn=200,n_sinkhorn_last=1000,n_lloyds=5,
        cost_function=None,cost_params={},
        s0=4,default_init=True):
        
        self.cost = cost_function
        self.cost_params = cost_params
        self.n_sinkhorn0 = n_sinkhorn
        self.n_sinkhorn = n_sinkhorn
        self.n_sinkhorn_last = n_sinkhorn_last
        self.n_lloyds = n_lloyds
        self.s0 = s0
        self.default_init = default_init
    
    def cost_matrix(self,data,masks=None):
        if self.cost is None:
            raise ValueError("A cost function is required")
        if isinstance(self.cost,list):
            assert isinstance(self.cost_params,list) and len(self.cost_params)==len(self.cost)
            costs = [cost(data,**params) for cost,params in zip(self.cost,self.cost_params)]
            return utils.multi_costs(costs,masks)
        else:
            return self.cost(data,**self.cost_params)
    
    def lloyd_step(self,
                   data,
                   cost_matrix=None,masks=None,
                   cap=None,
                   sinkhorn_algo=sinkhorn,
                   b=None,default_init=False,show_progress=False,
                   tau=0.0,to_bary=False,weight=None,**kwargs):
        
        cost, grad_cost = self.cost_matrix(data,masks=masks) if cost_matrix is None else cost_matrix
        if cap is not None:
            cost = cost.clamp(0.0,cap)
        scales = annealing_schedule(self.n_sinkhorn,M_grid=int(data.y.shape[0] ** (1/data.d)),s0=self.s0)
        sinkhorn_algo(cost,scales,
                      data.f_x,data.g_y,
                      a=data.volumes,b=b,default_init=default_init,show_progress=show_progress,**kwargs)
        
        self.Laguerre_allocation(data,cost)
        
        vols = data.allocation_matrix().sum(1).squeeze().reshape(data.N_crystals) * data.vol_grid
        vol_dev = (vols[:data.N_cells] - data.volumes[:data.N_cells]).abs()/data.volumes[:data.N_cells]
        vol_dev_max = vol_dev.max()
        vol_dev_mean = vol_dev.mean()
        print(f"Maximum volume deviation = {vol_dev_max}",flush=True)
        print(f"Mean volume deviation = {vol_dev_mean}",flush=True)
            
        return self.incompressibility_force(data,grad_cost,tau=tau,to_bary=to_bary,weight=weight)
        
    def solve(self,
              data,cost_matrix=None,masks=None,
              cap=None,
              sinkhorn_algo=sinkhorn,
              b=None,default_init=False,show_progress=False,
              tau=0.0,to_bary=False,weight=None,**kwargs):
        
        self.n_sinkhorn = self.n_sinkhorn0
        
        for it in range(self.n_lloyds):
            if it == self.n_lloyds - 1:
                self.n_sinkhorn = self.n_sinkhorn_last
                            
            F = self.lloyd_step(    
                data,
                cost_matrix=cost_matrix,masks=masks,
                cap=cap,
                sinkhorn_algo=sinkhorn_algo,
                b=b,default_init=default_init,show_progress=show_progress,
                tau=tau,to_bary=to_bary,weight=weight,**kwargs)
            
            data.x[:] = utils.apply_bc_inside(data.x+F,bc=data.bc,L=data.L)

    def Laguerre_allocation(self,data,cost_matrix):
        if len(data.f_x) == cost_matrix.shape[0]:
            mat = cost_matrix - LazyTensor(data.f_x[:,None,None])
            data.labels[:] = mat.argmin(axis=0).reshape(data.y.shape[0])
        else:
            mat = cost_matrix - LazyTensor(data.f_x[:-1,None,None])
            m, am = mat.min_argmin(axis=0)
            m = m.squeeze()
            am = am.squeeze().float()
            data.labels[m<=-data.f_x[-1]] = am[m<=-data.f_x[-1]]
            data.labels[m>-data.f_x[-1]] = data.x.shape[0] + 42.0
        
    def incompressibility_force(self,data,grad_cost,tau=0.0,to_bary=False,weight=None):
        if isinstance(tau,torch.Tensor):
            tau = tau.view(data.x.shape[0],1)
        A = data.allocation_matrix()
        if to_bary:
            if weight is None:
                XY = data.lazy_XY()
                weight = (grad_cost * XY).sum(-1) / ((XY ** 2).sum(-1) + 1e-8)
            bary = data.barycenters(weight=weight)
            return tau*(bary - data.x)
        else:
            F = (A * grad_cost).sum(1) * data.vol_grid
            return -tau*F