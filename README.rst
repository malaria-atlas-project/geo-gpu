Usage so far
============

The following code generates a distance matrix, overwrites it with a covariance
matrix and Cholesky factors that matrix.
::

    nbx = 40
    blocksize=16
    nby = 60
    
    x = np.arange(blocksize*nbx,dtype=d)
    y = np.arange(blocksize*nby,dtype=d)
    
    D = geo_gpu.CudaDistance(geo_gpu.euclidean, blocksize)
    C = geo_gpu.CudaRawCovariance(geo_gpu.exponential, blocksize)
    
    D_eval = D(x,x,symm=True)
    C_eval = C(D_eval, amp=2., scale=10., symm=True)
    S = geo_gpu.cholesky(C_eval, blocksize)    

``CudaDistance`` and ``CudaRawCovariance`` are subclasses of ``CudaMatrixFiller``.
Each time these objects are called with a new set of parameters (eg 
``amp=2., scale=10.``) they compile corresponding symmetric and nonsymmetric GPU 
kernels for the current datatype behind the scenes. They cache these kernels to 
avoid recompiling them for repeated calls. That means all parameter values are
compiled in as constants.

When called directly as in the example above, ``CudaDistance`` and ``CudaRawCovariance``
return numpy arrays. However, they each have a lower-lever method called ``gpu_call`` 
which returns a reference to an array on the GPU. These methods can optionally take 
GPU arrays as inputs. We can use these methods to pipe distance functions and
covariance functions together.

The function ``geo_gpu.cholesky`` similarly compiles and caches kernels for each
block-size and dtype behind the scenes. It also makes use of a lower-level 
function, ``geo_gpu.cholesky_gpu``, which will make it faster to evaluate and 
factorize covariance matrices when the actual evaluation is not needed.

Extending
=========

You can make a new distance like so:
::

    euclidean = {'name': 'euclidean',
    'preamble': "",
    'params':(),
    'body': """
        {{dtype}} d = 0;
        for(int i = 0; i < ndx; i++)
        {
            {{dtype}} dev = x[nxi+i] - y[nyj+i];
              d += dev*dev;
        }
        return sqrt(d);
    """}

or a new covariance function like so:
::

    exponential = {'name': 'exponential', 
    'preamble': "", 
    'params':('amp','scale'),
    'body': """
    d[0]=exp(-abs(d[0])/{{scale}})*{{amp}}*{{amp}};
    """}

``Params`` is a list of the name of the special parameters ``exponential`` takes, 
and ``body`` is a code snippet describing how to fill in a single element. The
body code may contain the parameters enclosed in double curly braces, and can
also use the template parameter ``{{dtype}}``.

The body code can also contain 'if' statements. The 'generic' covariance function 
template uses these to produce separate symmetric and nonsymmetric kernels:
::

    generic = """
    #define BLOCKSIZE {{blocksize}}

    {{preamble}}

    __device__ void {{funcname}}({{dtype}} *d)
    {
    {{body}}
    }
    __global__ void f({{dtype}} *cuda_matrix, int nx, int ny)
    {
    {{ if symm }}
    if(blockIdx.x >= blockIdx.y){ 
    {{ endif }}
    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
    {{funcname}}(cuda_matrix + nyj*nx + nxi);
    __syncthreads;
    {{if symm }}
    cuda_matrix[nxi*nx + nyj] = cuda_matrix[nyj*nx + nxi];
    }   {{ endif }}
    }"""
    
If ``symm`` is true, the stuff between the if blocks is kept; otherwise it's thrown out.