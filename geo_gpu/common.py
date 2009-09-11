import numpy as np

dtype_names = {
    np.dtype('float64'): 'double',
    np.dtype('float32'): 'float'}

# dist_generic has tokens:
#   - preamble, body, funcname : specialize to particular function
#   - dtype, blocksize : specialize further
dist_generic = """
#define BLOCKSIZE {{blocksize}}

{{preamble}}

__device__ {{dtype}} {{funcname}}({{dtype}} *x, {{dtype}} *y, int nxi, int nyj, int ndx)
{
{{body}}
}
__global__ void f({{dtype}} *cuda_matrix, {{dtype}} *x, {{dtype}} *y, int nx, int ndx)
{
    {{ if symm }}if(blockIdx.x >= blockIdx.y){ {{ endif }}
    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
    {{dtype}} d_xi_yj = {{funcname}}(x,y,nxi,nyj,ndx);
    __syncthreads;
    cuda_matrix[nyj*nx + nxi] = d_xi_yj;{{ if symm }}
    cuda_matrix[nxi*nx + nyj] = d_xi_yj;
} {{ endif }}
}
"""

# cov_generic has tokens:
#   - preamble, body, funcname : specialize to particular function
#   - dtype, blocksize : specialize further
cov_generic = """
#define BLOCKSIZE {{blocksize}}

{{preamble}}

__device__ {{dtype}} {{funcname}}({{dtype}} *d)
{
{{body}}
}
__global__ void f({{dtype}} *cuda_matrix, {{dtype}} *x, {{dtype}} *y, int nx, int ndx)
{
    {{ if symm }}if(blockIdx.x >= blockIdx.y){ {{ endif }}
    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
    {{dtype}} d_xi_yj = {{funcname}}(cuda_matrix + nyj*nx + nxi);
    __syncthreads;{{if symm }}
    cuda_matrix[nxi*nx + nyj] = d_xi_yj;
} {{ endif }}
}
"""