# Copyright (C) 2009  Bernhard Seiser and Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from cholesky import*
from cov_funs import *
from distances import *
from common import *
import pycuda as cuda
import pymc
import numpy as np


# TODO: Observations!
# TODO: Nuggets in Cholesky decomps
# TODO: Diagonal calls
class CudaCovariance(pymc.gp.Covariance):
    """A class mirroring pymc.gp.Covariance, but with all computations done on a GPU."""
    def __init__(self, distance, covariance):
        self.distance = distance
        self.covariance = covariance
        self.ndim = None
        self.observed = False
        
        if self.distance.dtype != self.covariance.dtype:
            raise ValueError, 'Distance function has dtype %s, but covariance function has dtype %s.'%(self.distance.dtype, self.covariance.dtype)

        self.dtype = self.covariance.dtype
        
    def gpu_call(self, x, y, symm=False):
        
        # Remember shape of x, and then 'regularize' it.
        orig_shape = np.shape(x)
        if len(orig_shape)>1:
            orig_shape = orig_shape[:-1]

        ndimx = x.shape[-1]
        lenx = x.shape[0]
        
        ndimy = y.shape[-1]
        leny = y.shape[0]

        if not ndimx==ndimy:
            raise ValueError, 'The last dimension of x and y (the number of spatial dimensions) must be the same.'
        

        # Safety
        if self.ndim is not None:
            if not self.ndim == ndimx:
                raise ValueError, "The number of spatial dimensions of x, "+\
                                    ndimx.__str__()+\
                                    ", does not match the number of spatial dimensions of the Covariance instance's base mesh, "+\
                                    self.ndim.__str__()+"."
        
        c_gpu = self.distance.gpu_call(x,y,symm)
        self.covariance.gpu_call(c_gpu,x.shape[0],y.shape[0],symm=False)
        return c_gpu
        
    def diag_gpu_call(self, x):
        raise NotImplemented
    
    def gpu_cholesky(self, x, blocksize, nugget=None):
        u_gpu = self.gpu_call(x, x)
        if nugget is not None:
            raise NotImplemented
        cholesky_gpu(u_gpu, x.shape[0], self.dtype, blocksize)
        return u_gpu
        
    def cholesky(self, x, observed=True, regularize=True, nugget=None, blocksize=16):
        x=pymc.gp.regularize_array(x)
        u_gpu = self.gpu_cholesky(x,blocksize,nugget)
        return gpu_to_ndarray(u_gpu, self.dtype, (x.shape[0],)*2)
        
    def continue_cholesky(self, x, x_old, u_gpu_old, observed=True, nugget=None):
        raise NotImplemented
        
    def __call__(self, x, y=None, observed=True, regularize=True):

        symm = y is x
        x=pymc.gp.regularize_array(x)

        # Diagonal case
        if y is None:
            v_gpu = self.diag_gpu_call(x)
            return gpu_to_ndarray(v_gpu, self.dtype, (x.shape[0]))

        # Full case
        else:                
            y = pymc.gp.regularize_array(y)
            c_gpu = self.gpu_call(x,y,symm=symm)
            return gpu_to_ndarray(c_gpu, self.dtype, (x.shape[0], y.shape[0]))
            
            
class CudaCudaCudaCudaCudaCovariance(CudaCovariance):
    """Krigin will be easy if you promise to like my streams..."""