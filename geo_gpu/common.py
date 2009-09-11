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
import pycuda.autoinit
from pycuda.driver import CompileError
import sys
from template import *

dtype_names = {
    np.dtype('float64'): 'double',
    np.dtype('float32'): 'float'}

class CudaMatrixFiller(object):
    """
    Base class for distance and covariance functions.
    """
    generic = None
    def __init__(self, body, blocksize):
        self.blocksize = blocksize
        self.name = body['name']
        self.source = templ_subs(self.generic, funcname=body['name'], preamble=body['preamble'], body=body['body'], blocksize=blocksize)
        self.params = body['params']
        self.sources = {}
        self.modules = {}
        
        for dtype in [np.dtype('float64'), np.dtype('float32')]:
            self.sources[dtype] = {}
            for symm in [True, False]:
                self.sources[dtype][symm] = templ_subs(self.source, symm=symm, dtype=dtype_names[dtype])
        
    def compile_with_parameters(self, **params):
        """
        Generates and compiles a CUDA module with a new set of parameters.
        If a kernel for the given parameters already exists, does nothing.
        """
        param_tup = tuple([params[k] for k in self.params])

        if not self.modules.has_key(param_tup):
            self.modules[param_tup] = {}
            for dtype in [np.dtype('float64'), np.dtype('float32')]:
                self.modules[param_tup][dtype] = {}
                for symm in [True, False]:
                    try:
                        s = templ_subs(self.sources[dtype][symm], **params)
                        self.modules[param_tup][dtype][symm] = cuda.SourceModule(s)
                    except CompileError:
                        cls, inst, tb = sys.exc_info()
                        new_msg = """ Failed to compile %s with dtype %s, symm=%s. Module source follows. 
NVCC's error message should be above the traceback.

%s 

Original error message from PyCuda: %s"""%(self.name, dtype, symm, add_line_numbers(s), inst.message)
                        raise cls, cls(new_msg), tb
                
        return self.modules[param_tup]