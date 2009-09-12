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
    def __init__(self, body, dtype, blocksize, **params):
        self.blocksize = blocksize
        self.__dict__.update(body)
        self.dtype = np.dtype(dtype)
        
        s = templ_subs(self.generic, funcname=body['name'], preamble=body['preamble'], body=body['body'])

        self.source = templ_subs(s, blocksize=blocksize, dtype=dtype_names[dtype], **params)
        self.sources = {}
        self.modules = {}

        for symm in [True, False]:
            try:
                self.sources[symm] = templ_subs(self.source, symm=symm)
                self.modules[symm] = cuda.SourceModule(self.sources[symm])
            except CompileError:
                cls, inst, tb = sys.exc_info()
                new_msg = """ Failed to compile %s with dtype %s, symm=%s. Module source follows. 
NVCC's error message should be above the traceback.

%s 

Original error message from PyCuda: %s"""%(self.name, dtype, symm, add_line_numbers(s), inst.message)
                raise cls, cls(new_msg), tb