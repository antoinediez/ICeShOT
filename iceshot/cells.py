import warnings
import math
import torch
from pykeops.torch import LazyTensor
from . import sample
from . import utils

class DataPoints:
    r"""Main class for seeds/source data points.
    
    :ivar device: default device for all the tensors (same as ``seeds`` device)
    :ivar dtype: data type
    :ivar d: dimenstion
    :ivar x: seeds positions
    :ivar y: source positions
    :ivar bc: boundary conditions
    :ivar L: box size as tensor
    """
    
    def __init__(self,seeds,source,bc=None,box_size=[1.0,1.0]):
        r"""Initialize

        Args:
            seeds ((N,d) Tensor): seed points
            source ((M,d) Tensor): source points
            bc (optional): boundary condition. Defaults to None.
            box_size (list, optional): Defaults to [1.0,1.0].
        """
        self.device = seeds.device
        self.dtype = seeds.dtype
        self.d = seeds.shape[1]
        self.x = seeds.detach().clone()    #(N_cells,d)
        self.y = source.detach().clone()
        self.bc = bc
        self.L = torch.tensor(box_size,device=self.device,dtype=self.dtype)
        
    def lazy_XY(self):
        r"""Return the XY matrix as a LazyTensor.
        
        The XY matrix is the matrix :math:`(y_j - x_i)_{ij}` where :math:`y` and :math:`x` are respectively the source and seed points.

        Returns:
            (N,M) LazyTensor
        """
        x_i = LazyTensor(self.x[:,None,:])  # (N, 1, D)
        y_j = LazyTensor(self.y[None,:,:])  # (1, M, D)
        return utils.apply_bc(y_j-x_i,bc=self.bc,L=self.L)

class Cells(DataPoints):
    """Main class for particles with a volume.
    
    :ivar N_cells: initial number of cells 
    :ivar N_crystals: current number of cells
    :ivar M_grid: number of source points
    :ivar vol_grid: volume of a source point (equal to 1/``M_grid``)
    :ivar axis: (N,d) Tensor of axes
    :ivar ar: (N,) Tensor of aspect ratios
    :ivar orientation: (N,d,d) Tensor of orientations
    :ivar f_x: (N,) Tensor of Kantorovich potentials
    :ivar g_y: (M,) Tensor of Kantorovich potentials
    :ivar labels: (M,) Tensor of labels
    """

    def __init__(self,
                 seeds,source,
                 vol_x,
                 extra_space=None,
                 fluid_size=None,
                 orientation=None,
                 axis=None,ar=None,
                 bc=None,
                 box_size=[1.0,1.0],
                 jct_method='Kmin'):
        """Initialize

        Args:
            seeds ((N,d) Tensor): seed points
            source ((M,d) Tensor): source points
            vol_x (N Tensor): volumes of the particles
            extra_space (str, optional): TBI. Defaults to None.
            fluid_size (float): TBI. Defaults to None.
            orientation ((N,d,d) Tensor, optional): Orientation matrices. Defaults to None (automatically computed from axis and aspect ratio).
            axis ((N,d) Tensor, optional): Orientation axis. Defaults to None (random axis).
            ar (N tensor, optional): Aspect ratios. Defaults to 1.
            bc (optional): boundary condition. Defaults to None.
            box_size (list, optional): Defaults to [1.0,1.0].
            jct_method (string, optional): The method used to compute junction points. Can be 'linear' (faster on CPU) or 'Kmin' (faster on GPU). Defaults to 'linear'. 
        """
                
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
        self.jct_method = jct_method
        
    @property
    def M_grid(self):
        return self.y.shape[0]
    
    @property
    def N_crystals(self):
        return self.x.shape[0]
    
    def orientation_from_axis(self):
        """Compute the orientation matrices from the axes and aspect ratios.

        Raises:
            NotImplementedError: Only in 2D and 3D

        Returns:
            Tensor: Orientation matrices
        """
        if self.d==2:
            th = torch.atan2(self.axis[:,1],self.axis[:,0])
            c = torch.cos(th)
            s = torch.sin(th)
            a11 = (1/self.ar*(c**2) + self.ar*(s**2)).reshape((self.N_crystals,1,1))
            a12 = ((1/self.ar - self.ar)*c*s).reshape((self.N_crystals,1,1))
            a21 = ((1/self.ar - self.ar)*c*s).reshape((self.N_crystals,1,1))
            a22 = (1/self.ar*(s**2) + self.ar*(c**2)).reshape((self.N_crystals,1,1))
            A = torch.cat((torch.cat((a11,a21),dim=1),torch.cat((a12,a22),dim=1)),dim=2)
        elif self.d==3:
            ar = self.ar.reshape((self.N_crystals,1,1))
            z = torch.zeros_like(ar)
            th = torch.acos(self.axis[:,2])
            ct = torch.cos(th).reshape((self.N_crystals,1,1))
            st = torch.sin(th).reshape((self.N_crystals,1,1))
            phi = torch.atan2(self.axis[:,1],self.axis[:,0])
            cp = torch.cos(phi).reshape((self.N_crystals,1,1))
            sp = torch.sin(phi).reshape((self.N_crystals,1,1))
            R = torch.cat((torch.cat((cp*st,-sp,-cp*ct),dim=2),
                           torch.cat((sp*st,cp,-sp*ct),dim=2),
                           torch.cat((ct,z,st),dim=2)),
                           dim=1)
            D = torch.cat((torch.cat((1.0/ar,z,z),dim=2),
                           torch.cat((z,ar**0.5,z),dim=2),
                           torch.cat((z,z,ar**0.5),dim=2)),
                           dim=1)
            A = torch.matmul(R,torch.matmul(D,torch.transpose(R,1,2)))
        else:
            raise NotImplementedError()
        return A
    
    def covariance_matrix(self,vols=None,bsr=False,**kwargs):
        """Compute the covariance matrices of a particle configuration.
        
        Args:
            vols (_type_, optional): Input volumes. Defaults to None (automatically computed).
            bsr (bool, optional): Use Block-Sparse-Reduction (faster). Defaults to False.

        Raises:
            NotImplementedError: Only in 2D and 3D
            
        Returns:
            (N,d,d) Tensor: Covariance matrix
        """

        center = self.barycenters(weight=1.0,bsr=bsr)
        
        if vols is None:
            vols = torch.bincount(self.labels.int())
        vols = vols[:self.N_cells].reshape((self.N_cells,1))
        
        if bsr:
            y, ranges_ij = self.sort_y() 
        
        X_i = LazyTensor(center[:,None,:])    # (N,1,D)
        Y_j = LazyTensor(self.y[None,:,:])    # (1,M,D)
        Z = utils.apply_bc(Y_j - X_i,bc=self.bc,L=self.L)    # (N,M,D)
        if self.d==2:
            cov_vec = LazyTensor.cat((Z[:,:,0]*Z, Z[:,:,1]*Z),-1)
        elif self.d==3:
            cov_vec = LazyTensor.cat((Z[:,:,0]*Z, Z[:,:,1]*Z, Z[:,:,2]*Z),-1)
        else:
            raise NotImplementedError()
        
        if bsr:
            cov_vec.ranges = ranges_ij
            cov = (cov_vec.sum(1)/(vols - 1)).reshape((self.N_crystals,self.d,self.d))
            self.y = y
        else:
            A = self.allocation_matrix()    #(N,M)
            cov = ((A * cov_vec).sum(1) / (vols - 1)).reshape((self.N_crystals,self.d,self.d))
        
        return cov
    
    def allocation_matrix(self,type="lazy"):
        """Compute the allocation matrix

        Args:
            type (str, optional): Output type. Can be "lazy" for LazyTensor or "dense" for Tensor. Defaults to "lazy".

        Returns:
            (N,M) Tensor or LazyTensor: The :math:`(i,j)` component of the allocation matrix is 1 if the source point :math:`j` belongs to the particle :math:`i` and 0 otherwise.
        """
        x = torch.arange(self.N_crystals,device=self.x.device,dtype=self.x.dtype)
        if type=="lazy":
            lazy_x = LazyTensor(x[:,None,None])
            lazy_labels = LazyTensor(self.labels[None,:,None].float())
            return (-(lazy_x - lazy_labels).abs()).step()
        elif type=="dense":
            return (-(x[:,None] - self.labels[None,:]).abs()).sign() + 1.0
        else:
            raise ValueError("Unknown type")
    
    def y_contact_matrix(self,r=None,bsr=False):
        
        y_i = LazyTensor(self.y[:,None,:])
        y_j = LazyTensor(self.y[None,:,:])
        XY = utils.apply_bc(y_j - y_i,self.bc,self.L)
        if bsr:
            if r is not None:
                raise NotImplementedError("1-pixel resolution only.")
            if self.bc is not None:
                raise NotImplementedError("Boudned box only.")
            r = 1.01 * math.sqrt(self.d) * ((1.0/self.M_grid) ** (1.0/self.d))
            matrix = ((r**2) - (XY **2).sum(-1)).step()
            i = torch.arange(self.M_grid,device=self.x.device,dtype=torch.int32).reshape((self.M_grid,1))
            ranges_i = torch.cat((i,i+1),dim=1)
            M = round((self.M_grid) ** (1/2)) if self.d==2 else round((self.M_grid) ** (1/3))
            redranges_j = torch.zeros((self.M_grid,2),dtype=torch.int32)
            redranges_j[i,0] = torch.maximum(i-M-2,torch.tensor(0))
            redranges_j[i,1] = torch.minimum(i+M+2,torch.tensor(self.M_grid-1))
            slices_i = (i+1).squeeze() 
            ranges_ij = (ranges_i,slices_i,redranges_j,ranges_i,slices_i,redranges_j)
            return matrix, ranges_ij
        else:
            if r is None:
                r = 1.01 * math.sqrt(self.d) * ((1.0/self.M_grid) ** (1.0/self.d))
            return ((r**2) - (XY **2).sum(-1)).step()
    
    def barycenters(self,weight=None,bsr=False):
        """Compute the (weighted) barycenter of each particle.

        Args:
            weight (float or Tensor, optional): Defaults to 1.
            bsr (bool, optional): Use Block-Sparse-Reduction (faster). Defaults to False.

        Returns:
            (N,d) Tensor: barycenters
        """
        if weight is None:
            weight = 1.0
        if isinstance(weight,torch.Tensor):
            weight = weight.view(self.x.shape[0],1,1)
        if bsr:
            y, ranges_ij = self.sort_y()
            XY = self.lazy_XY()
            XY.ranges = ranges_ij
            vols = torch.bincount(self.labels.int())
            bary = self.x + (weight*XY).sum(1)/(vols[:self.N_cells]+1e-8).view(self.x.shape[0],1)
            self.y = y
        else:
            A = self.allocation_matrix()
            XY = self.lazy_XY()
            bary = self.x + (weight*A*XY).sum(1)/((weight*A).sum(1)+1e-8).view(self.x.shape[0],1)
        return utils.apply_bc_inside(bary,bc=self.bc,L=self.L)
    
    def junctions(self,r=None,K=None):
        """Return a matrix of size MxK containing the cell labels of K neighbours of each pixel, padded with -1

        Args:
            r (float, optional): The 'vision radius' around each pixel point. By defaults, only the +/-1 neighbouring pixels are considered.
            K (int, optional): Must be larger than the number of pixels inside the 'vision radius'. By default, it is equal to 3**d + 1.

        Returns:
            (Tensor): A matrix of size MxK containing the cell labels of K neighbours of each pixel, padded with -1
        """
        if K is None:
            K = (3 ** self.d) + 1
        if self.jct_method=='Kmin':
            if r is None and self.bc is None:
                matrix, ranges_ij = self.y_contact_matrix(bsr=True)
                lab_neigh = matrix * LazyTensor(-(1.0 + self.labels[None,:,None]))
                lab_neigh.ranges = ranges_ij
            else:    
                lab_neigh = self.y_contact_matrix(r=r) * LazyTensor(-(1.0 + self.labels[None,:,None]))
            km = -lab_neigh.Kmin(K,dim=1)-1
            km = torch.cat((km,-torch.ones((self.M_grid,1))),dim=1) # Pad with -1. 
            _, indices = torch.unique_consecutive(km, return_inverse=True) # https://discuss.pytorch.org/t/best-way-to-run-unique-consecutive-on-certain-dimension/112662/3
            indices -= indices.min(dim=1, keepdims=True)[0]
            result = -torch.ones_like(km)
            return result.scatter_(1, indices, km)[:,:-1] # Sorted unique values
        elif self.jct_method=='linear':
            M = round((self.M_grid) ** (1/2)) if self.d==2 else round((self.M_grid) ** (1/3))
            output = -torch.ones((M,M,K)) if self.d==2 else -torch.ones((M,M,M,K))
            img = self.labels.reshape((M,M)) if self.d==2 else self.labels.reshape((M,M,M))
            for i in range(M):
                im = max(i-1,0)
                ip = min(i+1,M-1)
                for j in range(M):
                    jm = max(j-1,0)
                    jp = min(j+1,M-1)
                    if self.d==2:
                        unq = torch.unique(img[im:ip,jm:jp])
                        length = min(K,len(unq))
                        output[i,j,:length] = torch.flip(unq[:length],dims=(0,))
                    elif self.d==3:
                        for k in range(M):
                            km = max(k-1,0)
                            kp = min(k+1,M-1)
                            unq = torch.unique(img[im:ip,jm:jp,km:kp])
                            length = min(K,len(unq))
                            output[i,j,k,:length] = torch.flip(unq[:length],dims=(0,))
            return output.reshape((self.M_grid,K))
                    
            
                    
    
    def extract_boundary(self,r=None,K=None):
        """Return the indices of the grid-cells corresponding to a boundary pixel.
        """
        km = self.junctions(r=r,K=K)
        is_boundary = km[:,1]!=-1.0
        return is_boundary.nonzero().squeeze().long(), km[is_boundary.nonzero().squeeze(),:2].long()
        
    def triple_junctions(self,r=None,K=None):
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
    
    def sort_y(self):
        """Sort the source points according to their labels. 

        Returns:
            (Tensor, tuple): Positions before sorting and ``ranges_ij`` argument to be passed to the Block-Sparse-Reduction method. 
        """
        sorted_labels, sorting_indices = torch.sort(self.labels)
        y = self.y.detach().clone()
        y_sorted = y[sorting_indices]
        self.y = y_sorted
        diff_labels = sorted_labels[:-1] - sorted_labels[1:]
        end_j = (torch.nonzero(diff_labels) + 1).squeeze()
        
        i = torch.arange(self.N_crystals,device=self.x.device,dtype=end_j.dtype).reshape((self.N_crystals,1))
        ranges_i = torch.cat((i,i+1),dim=1)
        slices_i = (i+1).squeeze()
        redranges_i = torch.zeros((len(end_j),2),device=end_j.device,dtype=end_j.dtype)
        redranges_i[1:,0] = end_j[:-1]
        redranges_i[:,1] = end_j
        
        j = torch.arange(self.M_grid,device=self.x.device,dtype=end_j.dtype).reshape((self.M_grid,1))
        ranges_j = torch.cat((j,j+1),dim=1)
        slices_j = (j+1).squeeze()
        
        ranges_ij = ranges_i.type(torch.int32), slices_i.type(torch.int32), redranges_i.type(torch.int32), ranges_j.type(torch.int32), slices_j.type(torch.int32), ranges_i.type(torch.int32)
        
        if self.x.device=="cuda":
            torch.cuda.synchronize()
        
        return y, ranges_ij