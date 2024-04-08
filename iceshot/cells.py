import warnings
import math
import torch
from pykeops.torch import LazyTensor
from . import sample
from . import utils

class DataPoints:
    
    def __init__(self,seeds,source,bc=None,box_size=[1.0,1.0]):
        self.device = seeds.device
        self.dtype = seeds.dtype
        self.d = seeds.shape[1]
        self.x = seeds.detach().clone()    #(N_cells,d)
        self.y = source.detach().clone()
        self.bc = bc
        self.L = torch.tensor(box_size,device=self.device,dtype=self.dtype)
        
    def lazy_XY(self):
        x_i = LazyTensor(self.x[:,None,:])  # (N, 1, D)
        y_j = LazyTensor(self.y[None,:,:])  # (1, M, D)
        return utils.apply_bc(y_j-x_i,bc=self.bc,L=self.L)

class Cells(DataPoints):

    def __init__(self,
                 seeds,source,
                 vol_x,
                 extra_space=None,
                 fluid_size=None,
                 orientation=None,
                 axis=None,ar=None,
                 bc=None,
                 box_size=[1.0,1.0]):
                
        DataPoints.__init__(self,seeds,source,bc=bc,box_size=box_size)
        self.N_cells = seeds.shape[0]
        self.vol_grid = 1.0/self.M_grid   # float
            
        if extra_space is None:
            if vol_x.sum() != 1.0:
                warnings.warn("The input total volume has been renormalized to 1")
            self.volumes = vol_x/vol_x.sum()
        else:
            extra_vol = 1.0 - vol_x.sum().item()
            if extra_vol < 0.0:
                raise ValueError("The input total volume should be smaller than 1")
            
            if extra_space == "void":
                # self.x = torch.cat((self.x,torch.ones((1,self.d),dtype=self.dtype,device=self.device)),dim=0)
                self.volumes = torch.cat((vol_x,torch.tensor([extra_vol],dtype=self.dtype,device=self.device)))
            elif extra_space == "fluid":
                raise NotImplementedError()
                # if fluid_size is None:
                #     fluid_size = self.vol_grid
                # self.N_void = int(extra_vol/fluid_size)
                # self.vol_void = extra_vol/self.N_void
                # self.x_void = sample.sample_uniform(self.N_void,self.d,device=self.device,dtype=self.dtype)
                # self.x = torch.cat((self.x,self.x_void),dim=0)
                # self.volumes = torch.cat((vol_x,self.vol_void*torch.ones(self.N_void,device=self.device,dtype=self.dtype)))
                    
        self.R_mean = (self.volumes[:self.N_cells]/math.pi).sqrt().mean().item() if self.d==2 else ((self.volumes[:self.N_cells]/(4./3.*math.pi))**(1./3.)).mean().item()        

        if axis is None:
            axis = torch.randn((self.N_crystals,self.d),device=self.device,dtype=self.dtype)
            self.axis = axis/torch.norm(axis,dim=1).reshape((self.N_crystals,1))
        else:
            self.axis = axis
        if ar is None:
            self.ar = torch.ones(self.N_crystals,device=self.device,dtype=self.dtype)
        else:
            self.ar = ar
        if orientation is None:
            self.orientation = self.orientation_from_axis()
        else:
            self.orientation = orientation
                        
        self.f_x = torch.zeros_like(self.volumes)
        self.g_y = torch.zeros(self.M_grid,device=self.device,dtype=self.dtype)
        self.labels = torch.zeros(self.M_grid,device=self.device,dtype=self.dtype)
        
    @property
    def M_grid(self):
        return self.y.shape[0]
    
    @property
    def N_crystals(self):
        return self.x.shape[0]
    
    def orientation_from_axis(self):
        th = torch.atan2(self.axis[:,1],self.axis[:,0])
        c = torch.cos(th)
        s = torch.sin(th)
        a11 = (1/self.ar*(c**2) + self.ar*(s**2)).reshape((self.N_crystals,1,1))
        a12 = ((1/self.ar - self.ar)*c*s).reshape((self.N_crystals,1,1))
        a21 = ((1/self.ar - self.ar)*c*s).reshape((self.N_crystals,1,1))
        a22 = (1/self.ar*(s**2) + self.ar*(c**2)).reshape((self.N_crystals,1,1))
        A = torch.cat((torch.cat((a11,a21),dim=1),torch.cat((a12,a22),dim=1)),dim=2)
        return A
    
    def covariance_matrix(self,vols=None,**kwargs):
        A = self.allocation_matrix()    #(N,M)
        if vols is None:
            vols = A.sum(1)
        vols = vols.reshape((self.N_crystals,1))
        XY = self.lazy_XY()
        center = utils.apply_bc_inside(self.x + (A * XY).sum(1)/vols,bc=self.bc,L=self.L)
        X_i = LazyTensor(center[:,None,:])    # (N,1,D)
        Y_j = LazyTensor(self.y[None,:,:])    # (1,M,D)
        Z = utils.apply_bc(Y_j - X_i,bc=self.bc,L=self.L)    # (N,M,D)
        if self.d==2:
            cov_vec = LazyTensor.cat((Z[:,:,0]*Z, Z[:,:,1]*Z),-1)
        elif self.d==3:
            cov_vec = LazyTensor.cat((Z[:,:,0]*Z, Z[:,:,1]*Z, Z[:,:,2]*Z),-1)
        else:
            raise NotImplementedError()
        cov = ((A * cov_vec).sum(1) / (vols - 1)).reshape((self.N_crystals,self.d,self.d))
        return cov
    
    def allocation_matrix(self,type="lazy"):
        x = torch.arange(self.N_crystals,device=self.x.device,dtype=self.x.dtype)
        if type=="lazy":
            lazy_x = LazyTensor(x[:,None,None])
            lazy_labels = LazyTensor(self.labels[None,:,None].float())
            return (-(lazy_x - lazy_labels).abs()).step()
        elif type=="dense":
            return (-(x[:,None] - self.labels[None,:]).abs()).sign() + 1.0
        else:
            raise ValueError("Unknown type")
    
    def y_contact_matrix(self,r=None):
        if r is None:
            r = 1.1*(1.0/self.M_grid) ** (1.0/self.d)
        y_i = LazyTensor(self.y[:,None,:])
        y_j = LazyTensor(self.y[None,:,:])
        XY = utils.apply_bc(y_j - y_i,self.bc,self.L)
        return ((r**2) - (XY **2).sum(-1)).step()
    
    def barycenters(self,weight=None):
        if weight is None:
            weight = 1.0
        if isinstance(weight,torch.Tensor):
            weight = weight.view(self.x.shape[0],1,1)
        A = self.allocation_matrix()
        XY = self.lazy_XY()
        bary = self.x + (weight*A*XY).sum(1)/((weight*A).sum(1)+1e-8).view(self.x.shape[0],1)
        return bary
    
    def junctions(self,r=None,K=10):
        # Return a matrix of size MxK containing the cell labels of K neighbours of each pixel
        lab_neigh = self.y_contact_matrix(r=r) * LazyTensor(-(1.0 + self.labels[None,:,None]))
        km = -lab_neigh.Kmin(K,dim=1)-1
        _, indices = torch.unique_consecutive(km, return_inverse=True)
        indices -= indices.min(dim=1, keepdims=True)[0]
        result = -torch.ones_like(km)
        return result.scatter_(1, indices, km) #Sorted unique values
    
    def extract_boundary(self,r=None,K=10):
        # Return the indeices of the grid-cells corresponding to a boundary pixel
        km = self.junctions(r=r,K=K)
        is_boundary = km[:,1]!=-1.0
        return is_boundary.nonzero().squeeze().long(), km[is_boundary.nonzero().squeeze(),:2].long()
        
    def triple_junctions(self,r=None,K=10):
        # Return the list of indices of the grid-cells corresponding to triple junctions and the matrix of indices of corresponding particles
        km = self.junctions(r=r,K=K)
        is_triple = km[:,2]!=-1.0
        return is_triple.nonzero().squeeze().long(), km[is_triple.nonzero().squeeze(),:3].long()
    
    def clusterized_triple_junctions(self,r0=None,r=None):
        if r is None:
            r = 1.3*(1.0/self.M_grid) ** (1.0/self.d)
        ind_y,ind_x = self.triple_junctions(r=r0)
        y_junction = self.y[ind_y,:]
        y_junction_i = LazyTensor(y_junction[:,None,:])
        y_junction_j = LazyTensor(y_junction[None,:,:])
        is_close = (r**2 - ((y_junction_j - y_junction_i)**2).sum(-1)).step()
        ind_x_i = LazyTensor(ind_x[:,None,:].float())
        ind_x_j = LazyTensor(ind_x[None,:,:].float())
        same_ind = 1.0 - (ind_x_j - ind_x_i).abs().sum(-1).sign()
        same_cluster = is_close * same_ind
        cluster_pos = (same_cluster @ y_junction)/same_cluster.sum(1)
        uq_output = utils.unique(cluster_pos)
        return uq_output[0], ind_x[uq_output[-1],:]    
    
