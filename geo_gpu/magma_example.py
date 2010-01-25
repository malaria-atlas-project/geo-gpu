import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from magma_wrapper import magma_spotrf_gpu_wrap, magma_get_spotrf_nb_wrap
    
n = 1000

# Create matrix to be factored
A = np.eye(n, dtype='float32') + np.ones((n,n))*.1
A_gpu = cuda.to_device(A)

# Allocate pagelocked work array
nwork = magma_get_spotrf_nb_wrap(n)
work_gpu = cuda.pagelocked_empty((nb,nb), dtype='float32')

# Do Cholesky factorization
info = magma_spotrf_gpu_wrap('U', n, A_gpu, n, work_gpu)

# Copy back the Cholesy factor and check for correctness
L = cuda.from_device(A_gpu, (n,n), 'float32')
print np.abs(np.dot(L,L.T)-A).max()

