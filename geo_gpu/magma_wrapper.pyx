import pycuda.driver as cuda


# Will need to put magma.h on header search path for this to work
cdef extern from "magma.h":
    # Defined in magma_0.2/include/magma.h
    int magma_spotrf_gpu(char *uplo, int *n, float *a, int *lda, float *work, int *info)
    # Defined in magma_0.2/include/auxiliary.h
    int magma_get_spotrf_nb(int m)

def get_gpu_pointer(x):
    """
    Gets a pointer to the memory on the GPU. Works whether x
    is pagelocked host memory or GPU memory.
    """
    # See http://documen.tician.de/pycuda/driver.html#pycuda.driver.HostAllocation.get_device_pointer
    if isinstance(x,cuda.HostAllocation):
        return x.get_device_pointer()
    # See http://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
    elif isinstance(x, cuda.DeviceAllocation):
        return int(x)
    else:
        raise TypeError

def magma_get_spotrf_nb_wrap(m):
    """
    Figures out how big the work array needs to be
    """
    cdef int m_
    cdef int nb_

    m_ = m
    nb_ = magma_get_spotrf_nb(m_)
    
    return nb_
        
def magma_spotrf_gpu_wrap(uplo, n, A, lda, work):
    """
    Symmetric positive definite matrix A (on the gpu) wil be overwritten with its 
    Cholesky factorization
    """
    cdef int info_, n_, lda_, a_ptr_, work_ptr_
    cdef char uplo_
    cdef float *A_, *work_
    
    # Get memory addresses on GPU as integers
    a_ptr_ = get_gpu_pointer(A)
    work_ptr_ = get_gpu_pointer(work)
    
    # This is a little sketchy. Can you cast integers giving memory addresses
    # to float* in C?
    A_ = <float*> a_ptr_
    work_ = <float*> work_ptr_
    
    # Cast the Python arguments to their corresponding C types,
    # because MAGMA will be expecting pointers to them.
    n_ = n
    lda_ = lda
    uplo_ = uplo
    
    info_ = magma_spotrf_gpu(&uplo_, &n_, A_, &lda_, work_, &info_)
    
    return info_