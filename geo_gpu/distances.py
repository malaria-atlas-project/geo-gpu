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
import numpy
import warnings
import sys
from common import *
from template import *
import re

__all__ = ['cuda_distance','euclidean']

class cuda_distance(object):
    
    def __init__(self, body, blocksize):
        self.source = templ_subs(dist_generic, funcname=body['name'], preamble=body['preamble'], body=body['body'], blocksize=blocksize)
        self.params = body['params']
        self.sources = {}
        self.modules = {}
        
        for dtype in [numpy.dtype('float64'), numpy.dtype('float32')]:
            self.sources[dtype] = {}
            for symm in [True, False]:
                self.sources[dtype][symm] = templ_subs(self.source, symm=symm, dtype=dtype_names[dtype])
        
    def compile_with_parameters(self, *params):
        """Generates and compiles a CUDA module with a new set of parameters."""
        param_dict = dict(zip(self.params, params))
        self.modules[params] = {}
        for dtype in [numpy.dtype('float64'), numpy.dtype('float32')]:
            self.modules[params][dtype] = {}
            for symm in [True, False]:
                self.modules[params][dtype][symm] = cuda.SourceModule(templ_subs(self.sources[dtype][symm], **param_dict))
                
        return self.modules[params]
                
    def __call__(x,y,symm=False,dtype=numpy.dtype('float32'),**params):

        # Compile module if necessary
        param_tup = tuple([params[k] for k in self.params])
        if self.modules.has_key(param_tup):
            mod = self.modules[param_tup]
        else:
            mod = self.compile_with_parameters(param_tup)

        # (body, x, y, nx, ny, ndx, ndy, cmin, cmax, symm, dtype=numpy.dtype('float64'), blocksize=16):
        nx = x.shape[0]
        ny = y.shape[0]
        ndx = x.shape[1]
        ndy = ndx

        matrixBlocks = nx/blocksize

        #Load cuda function
        cuda_fct = mod.get_function("fillMatrix_euclidean_symmetric_full")

        #Allocate arrays on device
        x_gpu = cuda.mem_alloc(nx*ndx*dtype.itemsize)
        y_gpu = cuda.mem_alloc(nx*ndx*dtype.itemsize)
        matrix_gpu = cuda.mem_alloc(nx*nx*dtype.itemsize)

        #Copy memory from host to device
        cuda.memcpy_htod(x_gpu, x)
        cuda.memcpy_htod(y_gpu, y)

        #Convert input parameters
        nx = numpy.uint32(nx)
        ndx = numpy.uint32(ndx)

        #Execute cuda function
        cuda_fct(matrix_gpu, x_gpu, y_gpu, nx, ndx, block=(blocksize,blocksize,1), grid=(matrixBlocks,matrixBlocks))

        #Free memory on gpu
        x_gpu.free()
        y_gpu.free()

        #return matrix_gpu
        return matrix_gpu


euclidean = {'name': 'euclidean','preamble': '','body': """
    {{dtype}} d = 0;
    for(int i = 0; i < ndx; i++)
    {
        {{dtype}} dev = x[nxi+i] - y[nyj+i];
          d += dev*dev;
    }
    return sqrt(d);
""", 'params':()}