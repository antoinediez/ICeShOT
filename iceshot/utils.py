import os
import sys
import cv2
import numpy as np
import torch
from pykeops.torch import LazyTensor
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib import colors

def multi_costs(costs,masks):
    assert len(masks)==len(costs), f"`masks` must be a list of same length as `costs`"
    list_costs = [mult_tuple(mask[:,None,None],cost) for mask,cost in zip(masks,costs)]
    return sum([el[0] for el in list_costs]), sum([el[1] for el in list_costs])
    
def mult_tuple(s,t):
    return s*t[0],s*t[1]

def apply_bc_inside(X,bc=None,L=[1.0,1.0]):
    if bc is None:
        return X
    if bc=="periodic":
        return torch.remainder(X,1.0)
    elif bc=="periodic_rectangle":
        return torch.remainder(X,torch.tensor(L,dtype=X.dtype,device=X.device).view((1,X.shape[1])))

def apply_bc(X,bc=None,L=[1.0,1.0]):
    if bc is None:
        return X
    if isinstance(X, LazyTensor):
        if bc == "periodic":
            return X + (-0.5 - X).step() - (X - 0.5).step()
        elif bc == "periodic_rectangle":
            L_k = LazyTensor(L[None, None, :])  # (1,1,d)
            return X + (-L_k/2.0 - X).step() * L_k - (X - L_k/2.0).step() * L_k
    elif isinstance(X,torch.Tensor):
        if bc == "periodic":
            return X + ((-0.5 - X)>0).float() - ((X - 0.5)>0).float()
        elif bc == "periodic_rectangle":
            L_k = L[None, None, :]  # (1,1,d)
            return X + ((-L_k/2.0 - X)>0.0).float() * L_k - ((X - L_k/2.0)>0.0).float() * L_k
        
def unique(x, dim=0):
    # https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index

def log1pexp(x: torch.Tensor):
    """
    https://botorch.org/v/latest/api/_modules/botorch/utils/safe_math.html
    Numerically accurate evaluation of log(1 + exp(x)).
    See [Maechler2012accurate]_ for details.
    """
    mask = x <= 18
    return torch.where(
        mask,
        (lambda z: z.exp().log1p())(x.masked_fill(~mask, 0)),
        (lambda z: z + (-z).exp())(x.masked_fill(mask, 0)),
    )
    
def mkdir_name(name):
    dir_name = name
    exists = os.path.isdir(dir_name)
    k = 0
    while exists:
        k += 1
        dir_name = dir_name + "_" + str(k)
        exists = os.path.isdir(dir_name)
    return dir_name

def make_video(number_of_frames=None,prefix="t_",simu_name="simu",video_name='funny_video',rate=40):
    img_array = []
    current_directory = os.getcwd()
    frame_directory = current_directory+"/"+simu_name+"/frames"
    if number_of_frames is None:
        number_of_frames = len([name for name in os.listdir(frame_directory) if os.path.isfile(os.path.join(frame_directory, name))])
    for count in range(number_of_frames):
        filename = frame_directory+"/"+prefix+str(count)+'.png'
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(simu_name+"/"+video_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), rate,size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
def CustomCmap(from_rgb,to_rgb):
    # https://stackoverflow.com/questions/16267143/matplotlib-single-colored-colormap-with-saturation
    
    r1,g1,b1 = from_rgb if isinstance(from_rgb,tuple) else colors.to_rgb(from_rgb)
    r2,g2,b2 = to_rgb if isinstance(to_rgb,tuple) else colors.to_rgb(to_rgb)

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

def cmap_from_list(*numbers,color_names,alpha=None):
    Ns = np.array(numbers)
    N = int(Ns.sum())
    vals = np.zeros((N,4))
    for i in range(len(Ns)):
        vals[int(Ns[:i].sum()):int(Ns[:(i+1)].sum())] = colors.to_rgba(color_names[i])
    if alpha is not None:
        vals[:,3] = alpha   
    return ListedColormap(vals, name='custom')

def data_to_polar(data):
    if data.d != 2:
        raise NotImplementedError("Dimension 2 only...")
    XY = data.lazy_XY()
    r_ij = (XY ** 2).sum(-1).sqrt()
    th = torch.atan2(data.axis[:,1],data.axis[:,0])
    th_i = LazyTensor(th[:,None,None])
    th_ij = XY[:,:,1].atan2(XY[:,:,0]) - th_i
    return r_ij, th_ij