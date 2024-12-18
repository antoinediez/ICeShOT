import numpy as np 
import torch
from pykeops.torch import LazyTensor
from tqdm import tqdm
from . import utils

def LBFGSB(C_xy,scales,f_x,g_y,a=None,b=None,default_init=False,show_progress=False,max_epoch=10,tol=0.01,stopping_criterion="average",**kwargs):
    """LBFGS-B solver: update the potentials ``f_x`` and ``g_y``. 

    Args:
        C_xy ((N,M) LazyTensor): cost matrix
        scales (Tensor): The number of iterations is the length of ``scales``
        f_x ((N+1,) Tensor): Kantorovich potentials
        g_y ((M,) Tensor): Kantorovich potentials
        a ((N+1,) Tensor, optional): Defaults to 1.
        b ((M, ) Tensor, optional): Defaults to 1.
        default_init (bool, optional): Initial point for the solver: 0 (True) or ``f_x`` (False). Defaults to True.
        max_epoch (int, optional): Defaults to 10.
        tol (float, optional): Tolerance. Defaults to 0.01.
        stopping_criterion (str, optional): Can be "maximum" or "average". Defaults to "average".

    """
    
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
        g_y, yi = mat.min_argmin(axis=0)
        g_y = g_y.reshape(M)
        g_y = torch.min(g_y,torch.zeros_like(g_y))
        yi = yi.reshape(M)
        yi[g_y==0.0] = N
        res = (a * f_x).sum() + (b * g_y).sum()
        return res, yi

    seed_potentials = torch.zeros_like(a[:-1]) if default_init else f_x[:-1]
    optimizer = torch.optim.LBFGS(
        [seed_potentials], max_iter=n_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss, yi = dual_cost(seed_potentials)
        volumes = torch.bincount(yi,minlength=N+1)[:-1] / M
        seed_potentials.grad = volumes - a[:-1]
        return -loss
    
    for epoch in range(max_epoch):
        print(f"\nepoch = {epoch}",flush=True)
        optimizer.step(closure)

        # Compute the volume deviation 
        _, yi = dual_cost(seed_potentials)
        vols = torch.bincount(yi,minlength=N+1)[:-1] / M
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
    """Sinkhorn algorithm: update the potentials ``f_x`` and ``g_y``

    Args:
        C_xy ((N,M) LazyTensor): Cost matrix
        scales (Tensore): Annealing schedule
        f_x ((N,) Tensor): Kantorovich potential
        g_y ((M,) Tensor): Kantorovich potential
        a (_type_, optional): Defaults to 1/N
        b (_type_, optional): Defaults to 1/M
        default_init (bool, optional): Set the potential values to a good initial value. Defaults to True.
        show_progress (bool, optional): Defaults to False.
    """
    
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
    """Sinkhorn algorithm with an extra void particle: update the potentials ``f_x`` and ``g_y``

    Args:
        C_xy ((N,M) LazyTensor): Cost matrix
        scales (Tensore): Annealing schedule
        f_x ((N+1,) Tensor): Kantorovich potential
        g_y ((M,) Tensor): Kantorovich potential
        a (_type_, optional): Defaults to 1/N
        b (_type_, optional): Defaults to 1/M
        default_init (bool, optional): Set the potential values to a good initial value. Defaults to True.
        show_progress (bool, optional): Defaults to False.
    """
    
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
    """Optimal Transport solver class.
    
        :ivar cost: Cost function
        :ivar cost_params: Parameters to be passed to the cost function
        :ivar n_sinkhorn0: Initial number of iterations in the Sinkhorn algorithm or maximal number of iterations per optimization step in the LBFGS-B algorithm.
        :ivar n_sinkhorn: Number of iterations in the Sinkhorn algorithm or maximal number of iterations per optimization step in the LBFGS-B algorithm.
        :ivar n_sinkhorn: Final number of iterations in the Sinkhorn algorithm or maximal number of iterations per optimization step in the LBFGS-B algorithm.
    """
    
    def __init__(self,
        n_sinkhorn=200,n_sinkhorn_last=1000,n_lloyds=5,
        cost_function=None,cost_params={},
        s0=4,default_init=True):
        """Initialize

        Args:
            n_sinkhorn (int, optional): Number of iterations in the Sinkhorn algorithm or maximal number of iterations per optimization step in the LBFGS-B algorithm. Defaults to 200.
            n_sinkhorn_last (int, optional): ``n_sinkhorn`` at the last epoch. Defaults to 1000.
            n_lloyds (int, optional): Number of Lloyd steps. Defaults to 5.
            cost_function (fun, optional): Cost function. Defaults to None.
            cost_params (dict, optional): Parameters to be passed to the cost function. Defaults to {}.
            s0 (float, optional): Initial range of the annealing schedule. Defaults to 4.
            default_init (bool, optional): Defaults to True.
        """
        
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
                   tau=0.0,to_bary=False,weight=None,bsr=False,**kwargs):
        """Solve the OT problem and return the incompressibility forces for the current configuration (using the method ``incompressiblity_force``)

        Args:
            data (DataPoints)
            cost_matrix (fun, optional): Specify custom custom matrix. Defaults to None.
            masks (optional): Boolean masks for multi-cost cases. Defaults to None.
            cap (float, optional): Maximum value of the cost. Defaults to None.
            sinkhorn_algo (fun, optional): OT solver algorithm. Defaults to sinkhorn.
            b (Tensor, optional): To be passed to ``sinkhorn_algo``. Defaults to None.
            default_init (bool, optional): To be passed to ``sinkhorn_algo``. Defaults to False.
            show_progress (bool, optional): To be passed to ``sinkhorn_algo``. Defaults to False.
            tau (float, optional): Gradient-step (multiplicative factor) of the incompressibility force. Defaults to 0.0.
            to_bary (bool, optional): The incompressibility force is in the direction of the barycenters. Defaults to False.
            weight (float or Tensor, optional): Weights for the barycenters. Defaults to None.
            bsr (bool, optional): Use Block-Sparse-Reduction (faster). Defaults to False.

        Returns:
            (N,d) Tensor: incompressibility force
        """
        
        cost, grad_cost = self.cost_matrix(data,masks=masks) if cost_matrix is None else cost_matrix
        if cap is not None:
            cost = cost.clamp(0.0,cap)
        scales = annealing_schedule(self.n_sinkhorn,M_grid=int(data.y.shape[0] ** (1/data.d)),s0=self.s0,device=data.x.device)
        sinkhorn_algo(cost,scales,
                      data.f_x,data.g_y,
                      a=data.volumes,b=b,default_init=default_init,show_progress=show_progress,**kwargs)
        
        self.Laguerre_allocation(data,cost)
        
        vols = torch.bincount(data.labels.int()) * data.vol_grid
        vol_dev = (vols[:data.N_cells] - data.volumes[:data.N_cells]).abs()/data.volumes[:data.N_cells]
        vol_dev_max = vol_dev.max()
        vol_dev_mean = vol_dev.mean()
        print(f"Maximum volume deviation = {vol_dev_max}",flush=True)
        print(f"Mean volume deviation = {vol_dev_mean}",flush=True)
            
        return self.incompressibility_force(data,grad_cost,tau=tau,to_bary=to_bary,weight=weight,bsr=bsr,masks=masks)
        
    def solve(self,
              data,cost_matrix=None,masks=None,
              cap=None,
              sinkhorn_algo=sinkhorn,
              b=None,default_init=False,show_progress=False,
              tau=0.0,to_bary=False,weight=None,bsr=False,**kwargs):
        """Lloyd algorithm: perform ``self.n_lloyds`` times the operation ``data.x + F`` where ``F`` is the incompressibility force computed by `self.lloyd_step`.

        Args:
            data (DataPoints)
            cost_matrix (fun, optional): Specify custom custom matrix. Defaults to None.
            masks (optional): Boolean masks for multi-cost cases. Defaults to None.
            cap (float, optional): Maximum value of the cost. Defaults to None.
            sinkhorn_algo (fun, optional): OT solver algorithm. Defaults to sinkhorn.
            b (Tensor, optional): To be passed to ``sinkhorn_algo``. Defaults to None.
            default_init (bool, optional): To be passed to ``sinkhorn_algo``. Defaults to False.
            show_progress (bool, optional): To be passed to ``sinkhorn_algo``. Defaults to False.
            tau (float, optional): Gradient-step (multiplicative factor) of the incompressibility force. Defaults to 0.0.
            to_bary (bool, optional): The incompressibility force is in the direction of the barycenters. Defaults to False.
            weight (float or Tensor, optional): Weights for the barycenters. Defaults to None.
            bsr (bool, optional): Use Block-Sparse-Reduction (faster). Defaults to False.
        """
        
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
                tau=tau,to_bary=to_bary,weight=weight,bsr=bsr,**kwargs)
            
            data.x[:] = utils.apply_bc_inside(data.x+F,bc=data.bc,L=data.L)

    def Laguerre_allocation(self,data,cost_matrix):
        """Update the label of the source points given the Kantorovich potentials.

        Args:
            data (DataPoints)
            cost_matrix ((N,M) LazyTensor)
        """
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
        
    def incompressibility_force(self,data,grad_cost,tau=0.0,to_bary=False,weight=None,bsr=False,masks=None):
        """Compute the incompressibility force given a configuration and a cost. 

        Args:
            data (DataPoints)
            grad_cost ((N,M,d)): Gradient of the cost matrix
            tau (float, optional): Gradient-step (multiplicative factor) of the incompressibility force. Defaults to 0.0.
            to_bary (bool, optional): The incompressibility force is in the direction of the barycenters. Defaults to False.
            weight (float or Tensor, optional): Weights for the barycenters. Defaults to None.
            bsr (bool, optional): Use Block-Sparse-Reduction (faster). Defaults to False.
            masks (optional): Boolean masks for multi-costs situations. Defaults to None.

        Returns:
            (N,d) Tensor: Incompressibility force
        """
        if isinstance(tau,torch.Tensor):
            tau = tau.view(data.x.shape[0],1)
        A = data.allocation_matrix()
        if to_bary:
            if bsr:
                if weight is None:
                    y, ranges_ij = data.sort_y()
                    XY = data.lazy_XY()
                    XY.ranges = ranges_ij
                    _, grad_sorted = self.cost_matrix(data,masks=masks)
                    grad_sorted.ranges = ranges_ij
                    weight = (grad_sorted * XY).sum(-1) / ((XY ** 2).sum(-1) + 1e-8)
                bary = data.barycenters(weight=weight,bsr=True)
            else:
                if weight is None:
                    XY = data.lazy_XY()
                    weight = (grad_cost * XY).sum(-1) / ((XY ** 2).sum(-1) + 1e-8)
                bary = data.barycenters(weight=weight,bsr=False)
            return tau*(bary - data.x)
        else:
            if bsr:
                grad_sorted, y = self.compute_sorted_grad(data,masks=masks)
                F = grad_sorted.sum(1) * data.vol_grid
                data.y = y
            else:
                F = (A * grad_cost).sum(1) * data.vol_grid
            return -tau*F
        
    def compute_sorted_grad(self,data,masks=None):        
        y, ranges_ij = data.sort_y()
        _, grad_sorted = self.cost_matrix(data,masks=masks)
        grad_sorted.ranges = ranges_ij
        return grad_sorted, y