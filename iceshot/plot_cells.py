import matplotlib 
from matplotlib import pyplot as plt
import torch

class CellPlot:
    
    def __init__(self,crystals,cmap=plt.cm.bone,figsize=8,
                 plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=False,
                 scat_size=15,scat_color='red',
                 c_quiv='red',pivot="tail",
                 r=None,K=5,boundary_color='k',
                 plot_type="imshow",void_color=None,vmin0=0.00001,shrink_colors=True,M_grid=None):
        self.cmap = cmap
        self.shrink_colors = shrink_colors
        self.vmin0 = vmin0
        self.figsize = figsize
        
        self.plot_pixels = plot_pixels
        self.plot_scat = plot_scat
        self.plot_quiv = plot_quiv
        self.plot_boundary = plot_boundary
        
        self.void_color=void_color
        
        self.M_grid = int(crystals.M_grid ** (1/crystals.d)) if M_grid is None else M_grid
                
        self.fig, self.ax, self.plots = self.init_plot(
            crystals,
            plot_pixels=plot_pixels,plot_scat=plot_scat,plot_quiv=plot_quiv,plot_boundary=plot_boundary,
            scat_size=scat_size,scat_color=scat_color,
            c_quiv=c_quiv,pivot=pivot,
            r=r,K=K,boundary_color=boundary_color,
            plot_type=plot_type
            )
        
    def init_plot(self,crystals,
                  plot_pixels=True,plot_scat=True,plot_quiv=False,plot_boundary=False,
                  scat_size=15,scat_color='red',
                  c_quiv='red',pivot="tail",
                  r=None,K=5,boundary_color='k',
                  plot_type="imshow"):
        fig, ax = plt.subplots(figsize=(self.figsize,self.figsize))
        M_grid = self.M_grid
        ax.set_xlim(0,M_grid)
        ax.set_ylim(0,M_grid)
        plots = {}
        colors = self.colors(crystals)
        if self.void_color is not None:
            vmin = self.vmin0
            vmax = 1.0
            self.cmap.set_over(self.void_color)
            self.cmap.set_under(self.void_color)
        else:
            vmin = self.vmin0
            vmax = 1.0
        if plot_pixels:
            grid_colors = self.colors_to_grid(crystals,colors)
            if plot_type == "imshow":
                img = grid_colors.reshape(M_grid, M_grid)
                img = img.cpu().numpy()
                im = ax.imshow(img.transpose(),origin='lower',interpolation=None,cmap=self.cmap,vmin=vmin,vmax=vmax)
            elif plot_type == "scatter":
                if self.void_color is not None:
                    grid_colors[grid_colors==0.0] = -1.0
                im = ax.scatter(M_grid*crystals.y[:,0].cpu(),M_grid*crystals.y[:,1].cpu(),marker=",",s=2,c=grid_colors.cpu(),cmap=self.cmap,vmin=vmin,vmax=vmax)
            plots["pixels"] = im
        if plot_scat: 
            scat = ax.scatter(M_grid*crystals.x[:crystals.N_cells,0].cpu(),M_grid*crystals.x[:crystals.N_cells,1].cpu(),s=scat_size,c=scat_color)  
            plots["scat"] = scat
        if plot_quiv:
            v = crystals.axis[:crystals.N_cells]
            quiv = ax.quiver(M_grid*crystals.x[:crystals.N_cells,0].cpu(),M_grid*crystals.x[:crystals.N_cells,1].cpu(),v[:,0].cpu(),v[:,1].cpu(),color=c_quiv,pivot=pivot,zorder=2.5)
            plots["quiv"] = quiv
        if plot_boundary:
            index_bound, _ = crystals.extract_boundary(r=r,K=K)
            bnd = ax.scatter(M_grid*crystals.y[index_bound,0].cpu(),M_grid*crystals.y[index_bound,1].cpu(),marker=",",s=1,c=boundary_color)
            plots["boundary"] = bnd
        return fig, ax, plots
    
    def update_plot(self,crystals,r=None,K=3,shrink_colors=False):
        colors = colors = self.colors(crystals)
        M_grid = self.M_grid
        
        if self.plot_pixels:
            grid_colors = self.colors_to_grid(crystals,colors)
            if type(self.plots["pixels"]) == matplotlib.image.AxesImage:
                img = grid_colors.reshape(M_grid, M_grid)
                img = img.cpu().numpy()
                self.plots["pixels"].set_data(img.transpose())
            elif type(self.plots["pixels"]) == matplotlib.collections.PathCollection:
                if self.void_color is not None:
                    grid_colors[grid_colors==0.0] = -1.0
                self.plots["pixels"].set_color(self.cmap(grid_colors.cpu()))
                
        if self.plot_scat:
            self.plots["scat"].set_offsets(M_grid*crystals.x[:crystals.N_cells,:].cpu())
            
        if self.plot_quiv:
            self.plots["quiv"].set_offsets(M_grid*crystals.x[:crystals.N_cells,:].cpu())
            v = crystals.axis[:crystals.N_cells]
            self.plots["quiv"].set_UVC(v[:,0].cpu(),v[:,1].cpu())
            
        if self.plot_boundary:
            index_bound, _ = crystals.extract_boundary(r=r,K=K)
            self.plots["boundary"].set_offsets(M_grid*crystals.y[index_bound,:].cpu())
            
                    
    def colors(self,crystals):
        if self.shrink_colors:
            colors =  torch.linspace(self.vmin0+0.001,0.999,steps=crystals.N_cells,device=crystals.device,dtype=crystals.dtype)
            if crystals.N_crystals > crystals.N_cells:
                void_colors = 2.0 + 0.1 * torch.rand(crystals.N_crystals - crystals.N_cells,device=crystals.device,dtype=crystals.dtype) 
                colors = torch.cat((colors,void_colors))
        else:
            colors = torch.linspace(0.00001,1,steps=crystals.N_crystals,device=crystals.device,dtype=crystals.dtype)
        colors = colors.float()
        return colors
    
    def colors_to_grid(self,crystals,colors):
        A = crystals.allocation_matrix()
        return A.t() @ colors