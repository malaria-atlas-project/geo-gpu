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

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.driver import CompileError
import sys
from template import *

__all__ = ['dtype_names', 'substitute_dtypes', 'gpu_to_ndarray', 'CudaMatrixFiller','ndarray_to_gpu']

dtype_names = {
    np.dtype('float64'): 'double',
    np.dtype('float32'): 'float'}

dtype_archs = {}
    
def substitute_dtypes(param_dtypes, params, dtype):
    out = {}
    for k,v in param_dtypes.iteritems():
        out[k] = '(%s) %s'%(templ_subs(param_dtypes[k], dtype=dtype), np.asscalar(np.asarray(params[k],dtype=dtype)))
    return out

def gpu_to_ndarray(a_gpu, dtype, shape):
    a_cpu = np.empty(shape,dtype=dtype,order='F')        
    cuda.memcpy_dtoh(a_cpu,a_gpu)
    a_gpu.free()
    return a_cpu
    
def ndarray_to_gpu(a_cpu):
    a_gpu = cuda.mem_alloc(int(np.prod(a_cpu.shape)*a_cpu.dtype.itemsize))
    cuda.memcpy_htod(a_gpu, a_cpu)
    return a_gpu
    

class CudaMatrixFiller(object):
    """
    Base class for distance and covariance functions.
    """
    generic = None
    def __init__(self, cuda_code, dtype, blocksize, **params):
        
        self.blocksize = blocksize
        self.__dict__.update(cuda_code)
        self.dtype = np.dtype(dtype)
        
        if self.dtype != np.dtype('float32'):
            raise NotImplementedError, 'We do not have double-precision working yet.'
        
        s = templ_subs(self.generic, preamble=cuda_code['preamble'], body=cuda_code['body'])
        sp = templ_subs(s, **substitute_dtypes(cuda_code['params'], params, dtype_names[self.dtype]))

        self.source = templ_subs(sp, blocksize=blocksize, dtype=dtype_names[self.dtype])
        self.sources = {}
        self.modules = {}

        for symm in [True, False]:
            try:
                self.sources[symm] = templ_subs(self.source, symm=symm)
                self.modules[symm] = SourceModule(self.sources[symm])
            except CompileError:
                cls, inst, tb = sys.exc_info()
                new_msg = """ Failed to compile with dtype %s, symm=%s. Module source follows. 
NVCC's error message should be above the traceback.

%s 

Original error message from PyCuda: %s"""%(self.dtype, symm, add_line_numbers(self.sources[symm]), inst.message)
                raise cls, cls(new_msg), tb