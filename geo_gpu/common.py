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
        
    def compile_with_parameters(self, *params):
        """Generates and compiles a CUDA module with a new set of parameters."""
        param_dict = dict(zip(self.params, params))
        self.modules[params] = {}
        for dtype in [np.dtype('float64'), np.dtype('float32')]:
            self.modules[params][dtype] = {}
            for symm in [True, False]:
                try:
                    s = templ_subs(self.sources[dtype][symm], **param_dict)
                    self.modules[params][dtype][symm] = cuda.SourceModule(s)
                except CompileError:
                    cls, inst, tb = sys.exc_info()
                    new_msg = """ Failed to compile %s with dtype %s, symm=%s. Module source follows. 
NVCC's error message should be above the traceback.

%s 
  
Original error message from PyCuda: %s                  
"""%(self.name, dtype, symm, add_line_numbers(s), inst.message)
                    raise cls, cls(new_msg), tb
                
        return self.modules[params]