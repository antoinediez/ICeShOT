import math
import torch
from pykeops.torch import LazyTensor
from . import utils

def l2_cost(data,scaling=None,R=1.0,C=1.0,**kwargs):
    if scaling=="level_set":
        # The cost is equal to `C` at distance `R`
        sc = C/R**2
    elif scaling=="volume":
        # The integral of the cost on a ball of radius `R` is equal to the volume of the ball times `C`
        sc = C / (data.d/(2+data.d) * (R**2))
    elif scaling=="constant":
        sc = C
    else:
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    XY = data.lazy_XY()
    cost_matrix = sc * (XY ** 2).sum(-1) # (N,M)
    gradient = - sc * 2.0 * XY
    return cost_matrix, gradient
    
def power_cost(data,p=2.0,scaling=None,R=1.0,C=1.0,**kwargs):
    if scaling=="level_set":
        # The cost is equal to `C` at distance `R`
        sc = C/R**p
    elif scaling=="volume":
        # The integral of the cost on a ball of radius `R` is equal to the volume of the ball times `C`
        sc = C / (data.d/(p+data.d) * (R**p))
    elif scaling=="constant":
        sc = C
    else:
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    p_ = LazyTensor(p[:,None,None]) if isinstance(p,torch.Tensor) else p
    XY = data.lazy_XY()
    cost_matrix = sc * ((XY ** 2).sum(-1)) ** (p_/2.0) # (N,M)
    gradient = -sc * p_ * (((XY ** 2).sum(-1) + 1e-8) ** (p_/2.0 - 1.0)) * XY
    return cost_matrix, gradient

def anisotropic_power_cost(data,p=2.0,scaling=None,b=1.0,ar=None,C=1.0,**kwargs):
    if ar is None:
        ar = data.ar
    if scaling=="level_set":
        # `c(x,y) = C` is the equation of an ellipse with aspect ratio `ar` and short axis `b`. 
        sc = (1.0/(ar * b**2))**(p/2.0) * C
    elif scaling=="volume":
        # The integral of the cost on an ellipse of aspect ratio `ar` and short axis `b` is equal to the volume of the ellipse times `C`
        vol = math.pi*ar*(b**2)
        sc = 1.0/(2.0*math.pi/(p+data.d) * (b*(ar**0.5))**(p+data.d)) * vol * C
    elif scaling=="constant":
        sc = C
    else:
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    p_ = LazyTensor(p[:,None,None]) if isinstance(p,torch.Tensor) else p
    XY = data.lazy_XY()
    A_i = LazyTensor(data.orientation.view(len(data.x), 1, data.d * data.d))  # (N, 1, D * D)
    cost_matrix = sc * (XY | A_i.matvecmult(XY)) ** (p_/2.0)  # (N, M) symbolic LazyTensor
    gradient = -sc * p_ * ((XY | A_i.matvecmult(XY)) + 1e-8) ** (p_/2.0 - 1.0) * A_i.matvecmult(XY)
    return cost_matrix,gradient

def relu_cost(data,scaling=None,r=0.5,slp=1.0,R=1.0,C=1.0,**kwargs):
    if scaling=="level-set":
        sc = 1.0/(slp*(R-r)) * C
    elif scaling=="volume":
        vol = math.pi*(R**2)
        sc = 1.0/(2*math.pi*slp*((R**3 - r**3)/3.0 - r/2.0*(R**2 - r**2))) * vol * C
    elif scaling=="constant":
        sc = C
    else:
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    XY = data.lazy_XY()
    r_ij = (XY ** 2).sum(-1).sqrt()
    cost_matrix = sc * (slp*(r_ij - r)).relu()
    gradient = - sc * (r_ij - r).step() * slp * XY/(r_ij + 1e-8)
    return cost_matrix,gradient

def spherocylinders_cost(data,p=2.0,b=1.0,scaling=None,C=1.0,**kwargs):
    if data.d != 2:
        raise NotImplementedError()
    r_i = LazyTensor(data.ar[:,None,None]) if isinstance(data.ar,torch.Tensor) else data.ar
    p_ = LazyTensor(p[:,None,None]) if isinstance(p,torch.Tensor) else p
    b_ = LazyTensor(b[:,None,None]) if isinstance(b,torch.Tensor) else b
    C_ = LazyTensor(C[:,None,None]) if isinstance(C,torch.Tensor) else C
    if scaling=="level-set":
         b_i = b_   # Short-axis level-set
         sc = C_
    elif scaling=="volume":
        b_i = ((C_*(math.pi + 4*(r_i-1))/(2*math.pi/(p_+2) + 4*(r_i-1/(p_+1)))) ** 0.5) * b_
        sc = 1.0
    elif scaling=="constant":
        b_i = b_
        sc = C_
    else:
        b_i = b_
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    XY = data.lazy_XY()
    th = torch.atan2(data.axis[:,1],data.axis[:,0])
    th_i = LazyTensor(th[:,None,None])
    RXY_x = th_i.cos()*XY[:,:,0] + th_i.sin()*XY[:,:,1]
    RXY_y = -th_i.sin()*XY[:,:,0] + th_i.cos()*XY[:,:,1]
    cost_2 = (RXY_x.abs()/b_i - (r_i - 1)).relu()**2 + (RXY_y/b_i)**2
    cost_matrix = sc * cost_2 ** (p_/2.0)
    G_Rx = -LazyTensor(torch.cat((th.cos()[:,None,None],th.sin()[:,None,None]),dim=2))
    G_Ry = -LazyTensor(torch.cat((-th.sin()[:,None,None],th.cos()[:,None,None]),dim=2))
    G_cost_2 = 2.0/b_i*(RXY_x.abs()/b_i - (r_i-1)).relu()*RXY_x/(RXY_x.abs() + 1e-8)*G_Rx + 2*RXY_y*G_Ry/(b_i**2)
    gradient = sc * p_/2.0 * (cost_2 + 1e-8)**(p_/2.0 - 1) * G_cost_2
    return cost_matrix, gradient

def spherocylinders_2_cost(data,p=2.0,b=1.0,scaling=None,C=1.0,**kwargs):
    if data.d != 2:
        raise NotImplementedError()
    r_i = LazyTensor(data.ar[:,None,None]) if isinstance(data.ar,torch.Tensor) else data.ar
    p_ = LazyTensor(p[:,None,None]) if isinstance(p,torch.Tensor) else p
    b_ = LazyTensor(b[:,None,None]) if isinstance(b,torch.Tensor) else b
    C_ = LazyTensor(C[:,None,None]) if isinstance(C,torch.Tensor) else C
    if scaling=="level-set":
         b_i = b_   # Short-axis level-set
         sc = C_
    elif scaling=="volume":
        # The integral of the cost on the spherocylinder with short-axis b is the volume of this spherocylinder. 
        b_i = b_
        sc = (p_+2)/2.0 * C_
    elif scaling=="constant":
        b_i = b_
        sc = C_
    else:
        b_i = b_
        sc = 1.0
    if isinstance(sc,torch.Tensor):
        sc = sc.view(data.x.shape[0],1,1)
    
    XY = data.lazy_XY()
    th = torch.atan2(data.axis[:,1],data.axis[:,0])
    th_i = LazyTensor(th[:,None,None])
    RXY_x = th_i.cos()*XY[:,:,0] + th_i.sin()*XY[:,:,1]
    RXY_y = -th_i.sin()*XY[:,:,0] + th_i.cos()*XY[:,:,1]
    L0 = RXY_y.abs()
    L1 = (RXY_x**2 + RXY_y**2)/(RXY_x.abs()*(r_i-1) + (RXY_x**2 + RXY_y**2 - RXY_y**2*(r_i-1)**2).sqrt())
    cost0 = 1.0/b_i * (RXY_x.abs() - RXY_y.abs()*(r_i-1)).ifelse(L1,L0)
    cost_matrix = sc * (cost0 ** p_)

    G_Rx = -LazyTensor(torch.cat((th.cos()[:,None,None],th.sin()[:,None,None]),dim=2))
    G_Ry = -LazyTensor(torch.cat((-th.sin()[:,None,None],th.cos()[:,None,None]),dim=2))
    
    G_L0 = RXY_y/(RXY_y.abs()+1e-8) * G_Ry
    den = RXY_x.abs()*(r_i-1) + (RXY_x**2 + RXY_y**2 - RXY_y**2*(r_i-1)**2).sqrt()
    G_L1 = (2*RXY_x*G_Rx + 2*RXY_y*G_Ry)/den \
        -(RXY_x**2 + RXY_y**2)/den**2 * (\
            (r_i-1)*RXY_x/(RXY_x.abs()+1e-8)*G_Rx \
            + 1/(RXY_x**2 + RXY_y**2 - RXY_y**2*(r_i-1)**2).sqrt() * (RXY_x*G_Rx + (2*r_i - r_i**2)*RXY_y*G_Ry)
            )
    
    gradient = sc/b_i * p_ * ((cost0+1e-8) ** (p_ - 1)) * (RXY_x.abs() - RXY_y.abs()*(r_i-1)).ifelse(G_L1,G_L0)

    return cost_matrix, gradient


def polar_curve_to_cost(data,fun_r,p=2.0,**kwargs):
    r_ij, th_ij = utils.data_to_polar(data)
    p_ = LazyTensor(p[:,None,None]) if isinstance(p,torch.Tensor) else p
    XY = data.lazy_XY()
    XY_p = LazyTensor.cat((-XY[:,:,1],XY[:,:,0]),dim=-1)
    r0, dr0 = fun_r(th_ij,**kwargs)
    cost0 = r_ij/r0
    cost = cost0 ** p_
    gradient = p_ * ((cost0 + 1e-8) ** (p_-1)) * 1/(r_ij * r0) * (-XY + 1/r0*dr0*XY_p)
    return cost, gradient

def spherocylinder_polar_curve(th,b,ar=1.0,C=1.0):
    b_i = LazyTensor(b[:,None,None]) if isinstance(b,torch.Tensor) else b
    r_i = LazyTensor(ar[:,None,None]) if isinstance(ar,torch.Tensor) else ar
    C_ = LazyTensor(C[:,None,None]) if isinstance(C,torch.Tensor) else C
    cond = th.cos().abs() - (r_i-1)*th.sin().abs()
    R0 = b_i/(th.sin().abs() + 1e-8)
    R1 = b_i * (th.cos().abs()*(r_i-1) + (1 - (th.sin()*(r_i-1))**2).sqrt())
    r0 = C_ * cond.ifelse(R1,R0)
    dR0 = -b_i/(th.sin()**2 + 1e-8) * th.sin().sign() * th.cos()
    dR1 = b_i * (-th.cos().sign()*th.sin()*(r_i - 1) - 1/(1 - (th.sin()*(r_i-1))**2).sqrt() * th.sin()*th.cos()*(r_i-1)**2)
    dr0 = C_ * cond.ifelse(dR1,dR0)
    return r0, dr0
