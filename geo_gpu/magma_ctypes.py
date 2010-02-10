import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes

magma_path = '/home/malaria-atlas-project/sources/magma_0.2s/lib/objectfiles/libmagma.so'
libmagma = ctypes.cdll.LoadLibrary(magma_path)

int_pointer = lambda x: ctypes.pointer(ctypes.c_int(x))
char_pointer = lambda x: ctypes.pointer(ctypes.c_char(x))

_magma_spotrf_gpu = libmagma.magma_spotrf_gpu     
_magma_spotrf_gpu.restype = ctypes.c_uint
_magma_spotrf_gpu.argtypes = [ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_int),
                                np.ctypeslib.ndpointer(dtype='float32'),
                                ctypes.POINTER(ctypes.c_int)]

magma_get_spotrf_nb = libmagma.magma_get_spotrf_nb
magma_get_spotrf_nb.restype = ctypes.c_uint
magma_get_spotrf_nb.argtypes = [ctypes.POINTER(ctypes.c_int)]

def get_gpu_pointer(x):
    """
    Gets a pointer to the memory on the GPU. Works whether x
    is pagelocked host memory or GPU memory.
    """
    # See http://documen.tician.de/pycuda/driver.html#pycuda.driver.HostAllocation.get_device_pointer
    if isinstance(getattr(x,'base',None),cuda.HostAllocation):
        return x.base.get_device_pointer()
    # See http://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
    elif isinstance(x, cuda.DeviceAllocation):
        return int(x)
    else:
        raise TypeError

def magma_spotrf_gpu(uplo, n, A, lda, work):
    info = 1
    from IPython.Debugger import Pdb
    Pdb(color_scheme='LightBG').set_trace() 
    _magma_spotrf_gpu(char_pointer(uplo),
                        int_pointer(n),
                        # ====================================================================
                        # = I think this argument is responsible for the segmentation fault. =
                        # ====================================================================
                        ctypes.POINTER(ctypes.c_float).from_address(get_gpu_pointer(A)),
                        int_pointer(lda),
                        work.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        int_pointer(info))
    return info    

if __name__ == '__main__':
    n = 10
    
    # Create matrix to be factored
    A_orig = (np.eye(n) + np.ones((n,n))*.3).astype('float32')
    A_gpu = cuda.to_device(A_orig)

    # Allocate pagelocked work array
    nwork = magma_get_spotrf_nb(int_pointer(n))
    print nwork
    work_gpu = cuda.pagelocked_empty((nwork,nwork), dtype='float32')
    
    # # Do Cholesky factorization
    info = magma_spotrf_gpu('L', n, A_gpu, n, work_gpu)