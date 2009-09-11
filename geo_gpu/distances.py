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

import pycuda.driver as cuda
import pycuda.autoinit
import pymc as pm
import numpy
import warnings
import sys
from common import *
from template import *
import re

__all__ = ['CudaDistance','euclidean','dumb']

class CudaDistance(CudaMatrixFiller):
    """
    Wraps a CUDA distance function. Compiles it on-the-fly as needed.
    
    - Initialization: c = cuda_distance('euclidean', blocksize=16)
        Doesn't compile anything, but generates source templates for float and double,
        both symmetric and unsymmetric versions.
        
    - Calling: a = c(x,y,symm=False,dtype=numpy.dtype('float32'),**params)
        If 'params' is new (ie c has not yet been called with this set of parameters),
        then a new version of c is compiled. The parameters are 'compiled in' as
        constants.
        
        If 'params' has been seen before, the compiled module is retrieved from a
        cache.
        
        Either way, the compiled module is called and the result is copied to a
        NumPy array and returned.
        
    - On-GPU calling: a = c.gpu_call(...)
        Just like __call__, but the matrix is left alive on the GPU and is not
        copied into a numpy array. The on-GPU matrix is returned as an opaque
        object.
    """
    generic_tokens = ['preamble','body','funcname','blocksize','dtype']
    generic = """
#define BLOCKSIZE {{blocksize}}

{{preamble}}

__device__ {{dtype}} {{funcname}}({{dtype}} *x, {{dtype}} *y, int nxi, int nyj, int ndx)
{
{{body}}
}
__global__ void f({{dtype}} *cuda_matrix, {{dtype}} *x, {{dtype}} *y, int nx, int ndx)
{
    {{ if symm }}
    if(blockIdx.x >= blockIdx.y){ 
    {{ endif }}
    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
    {{dtype}} d_xi_yj = {{funcname}}(x,y,nxi,nyj,ndx);
    __syncthreads;
    cuda_matrix[nyj*nx + nxi] = d_xi_yj;
    {{ if symm }}
    cuda_matrix[nxi*nx + nyj] = d_xi_yj;
}   {{ endif }}
}
    """
        
    def __call__(self,x,y,symm=False,**params):
        """
        Deallocates all memory used on the GPU, returns result as NumPy matrix.
        The dtype of the return value will match that of the first argument,
        so be sure x.dtype is the dtype you want!
        """

        if x.dtype != y.dtype:
            raise ValueError, "Dtypes of arguments do not match. Sorry to be anal, but it's for your own good."

        dtype=x.dtype
        
        x = pm.gp.regularize_array(x).astype(x.dtype)
        y = pm.gp.regularize_array(y).astype(y.dtype)

        nx = x.shape[0]
        ny = y.shape[0]
        
        matrix_gpu = self.gpu_call(x,y,symm,dtype,**params)
        matrix_cpu = numpy.empty((nx,ny),dtype=dtype,order='F')
        cuda.memcpy_dtoh(matrix_cpu, matrix_gpu)
        matrix_gpu.free()
        
        return matrix_cpu.reshape(nx,ny)
        
    def gpu_call(self,x,y,symm=False,dtype=None,matrix_gpu=None,**params):
        """Leaves the generated matrix on the GPU, returns a PyCuda wrapper."""

        if dtype:
            x=x.astype(dtype)
            y=y.astype(dtype)
        else:
            dtype = x.dtype

        # Compile module if necessary
        param_tup = tuple([params[k] for k in self.params])
        if self.modules.has_key(param_tup):
            mod = self.modules[param_tup][dtype][symm]
        else:
            mod = self.compile_with_parameters(*param_tup)[dtype][symm]

        # (body, x, y, nx, ny, ndx, ndy, cmin, cmax, symm, dtype=numpy.dtype('float64'), blocksize=16):
        nx = x.shape[0]
        ny = y.shape[0]
        ndx = x.shape[1]
        ndy = ndx

        matrixBlocksx = nx/self.blocksize
        matrixBlocksy = ny/self.blocksize        

        #Load cuda function
        cuda_fct = mod.get_function("f")

        #Allocate arrays on device
        x_gpu = cuda.mem_alloc(nx*ndx*dtype.itemsize)
        y_gpu = cuda.mem_alloc(ny*ndy*dtype.itemsize)
        if matrix_gpu is None:
            matrix_gpu = cuda.mem_alloc(nx*ny*dtype.itemsize)

        #Copy memory from host to device
        cuda.memcpy_htod(x_gpu, x)
        cuda.memcpy_htod(y_gpu, y)

        #Convert input parameters
        nx = numpy.uint32(nx)
        ndx = numpy.uint32(ndx)

        #Execute cuda function
        cuda_fct(matrix_gpu, x_gpu, y_gpu, nx, ndx, block=(self.blocksize,self.blocksize,1), grid=(matrixBlocksx,matrixBlocksy))

        #Free memory on gpu
        x_gpu.free()
        y_gpu.free()

        #return matrix_gpu
        return matrix_gpu

dumb = {'name': 'dumb','preamble': '','params':(),
'body': """
    return ({{dtype}}) 1.0;
"""}

euclidean = {'name': 'euclidean','preamble': "",'params':(),
'body': """
    {{dtype}} d = 0;
    for(int i = 0; i < ndx; i++)
    {
        {{dtype}} dev = x[nxi+i] - y[nyj+i];
          d += dev*dev;
    }
    return sqrt(d);
"""}