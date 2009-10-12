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
from template import *
from common import *
import sys
import warnings

__all__ = ['cholesky','cholesky_gpu']

# Cholesky decomposition of NumPy matrix
def cholesky(C, blocksize=16):
    """
    Computes the Cholesky decomposition of C on the GPU. The dtype
    of the return value, and of the intermediate GPU data structures,
    will match that of C, so be sure C is of the dtype you want!
    """
    S = C.copy('F')
    
    nx = C.shape[0]
    
    S_gpu = cuda.mem_alloc(nx**2*S.dtype.itemsize)
    cuda.memcpy_htod(S_gpu,S)
    cholesky_gpu(S_gpu, nx, S.dtype, blocksize)
    cuda.memcpy_dtoh(S, S_gpu)
    S_gpu.free()
    
    warnings.warn('Zeroing lower triangle of S straight from Python, will be slow as molasses.')
    for i in xrange(nx):
        for j in xrange(i):
            S[i,j] = 0
    
    return S

cholesky_template = """
#define BLOCKSIZE {{blocksize}}
__global__ void d_choldc_topleft({{dtype}} *m, int matrix_size,  int boffset)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ {{dtype}} topleft[BLOCKSIZE][BLOCKSIZE+1];
    //topleft[ty][tx]=m[ty+BLOCKSIZE*boffset][tx+BLOCKSIZE*boffset];
    topleft[ty][tx]=m[(ty+BLOCKSIZE*boffset)*matrix_size + tx+BLOCKSIZE*boffset];
    __syncthreads();

    {{dtype}} fac;

    // in this loop tx labels column, ty row
    for(int k=0;k<BLOCKSIZE;k++)
    {
        __syncthreads();

        fac=rsqrtf(topleft[k][k]);
        __syncthreads();

        if ((ty==k)&&(tx>=k))
        {
            topleft[tx][ty]=(topleft[tx][ty])*fac;
        }
        __syncthreads();
  
        if ((ty>=tx)&&(tx>k))
        {
            topleft[ty][tx]=topleft[ty][tx]-topleft[tx][k]*topleft[ty][k];
        }
    }
    __syncthreads();

    if (ty>=tx) {
         //m[ty+BLOCKSIZE*boffset][tx+BLOCKSIZE*boffset]=topleft[ty][tx];
         m[(ty+BLOCKSIZE*boffset)*matrix_size + tx+BLOCKSIZE*boffset]=topleft[ty][tx];
    }
}

__global__ void d_choldc_strip({{dtype}} *m, int matrix_size,  int blockoffset)
{
    // +1 since blockoffset labels the "topleft" position and boff is the working position...
    int boffx = blockIdx.x+blockoffset+1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ {{dtype}} topleft[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ {{dtype}} workingmat[BLOCKSIZE][BLOCKSIZE+1];

    topleft[ty][tx] = m[(ty+blockoffset*BLOCKSIZE)*matrix_size +tx+blockoffset*BLOCKSIZE];
    workingmat[tx][ty] = m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];

    //topleft[ty][tx] = m[ty+blockoffset*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];
    //workingmat[tx][ty] = m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

    __syncthreads();

    if(ty==0)
    {
          for (int k=0;k<BLOCKSIZE;k++)
          {
            {{dtype}} dotprod=0.f;
            for (int m=0;m<k;m++)
            {
            dotprod+=topleft[k][m]*workingmat[m][tx];
            }
            workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleft[k][k];
          }
    }

    __syncthreads();

    //m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE] = workingmat[tx][ty];
    m[(ty+boffx*BLOCKSIZE)*matrix_size+tx+blockoffset*BLOCKSIZE] = workingmat[tx][ty];

}

__global__ void d_choldc_diagupdate({{dtype}} *m, int matrix_size,  int blockoffset)  
{
    int boffx = blockIdx.x+blockoffset+1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ {{dtype}} left[BLOCKSIZE][BLOCKSIZE+1];

    left[ty][tx]=m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];
    //left[ty][tx]=m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

    __syncthreads();

    {{dtype}} matrixprod=0.f;
    if(ty>=tx)  
    {
        for (int kk=0;kk<BLOCKSIZE;kk++)
        {
            matrixprod+=left[ty][kk]*left[tx][kk];
        }
        //m[ty+boffx*BLOCKSIZE][tx+boffx*BLOCKSIZE]-=matrixprod;
        m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+boffx*BLOCKSIZE]-=matrixprod;
    }
}

__global__ void d_choldc_loupdate({{dtype}} *m, int matrix_size,  int mat_blocks, int blockoffset)  
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int boffy=blockIdx.y+blockoffset+1;
    int boffx=boffy+1;

    __shared__ {{dtype}} left[BLOCKSIZE][BLOCKSIZE];

    __shared__ {{dtype}} upt[BLOCKSIZE][BLOCKSIZE+1];

    int tmpx,tmpy,tmpb;

    tmpy=__mul24(boffy,BLOCKSIZE);
    tmpb=__mul24(blockoffset,BLOCKSIZE);

    //upt[ty][tx]=m[ty+tmpy][tx+tmpb];
    upt[ty][tx]=m[(ty+tmpy)*matrix_size+tx+tmpb];

    for (;boffx<mat_blocks;boffx++){

        tmpx=__mul24(boffx,BLOCKSIZE);

        //left[ty][tx]=m[ty+tmpx][tx+tmpb];
        left[ty][tx]=m[(ty+tmpx)*matrix_size + tx+tmpb];

        __syncthreads();

        {{dtype}} matrixprod=0.f;

        // ty,tx'th thread works out (ty,tx) cmpt of the left-up product...
        for (int kk=0;kk<BLOCKSIZE;kk++)
        {
            matrixprod+=left[ty][kk]*upt[tx][kk];
        }

        __syncthreads();

        //m[ty+tmpx][tx+tmpy]-=matrixprod;
        m[(ty+tmpx)*matrix_size + tx + tmpy]-=matrixprod;
    }
}
"""

cholesky_modules = {}
cholesky_sources = {}

# Cholesky decomposition of matrixA which is already on gpu.
def cholesky_gpu(matrixA_gpu, matrixA_size, dtype, blocksize):

    if dtype_names[dtype] != 'float':
        raise NotImplementedError, 'Double precision not working yet.'

    # Compile a kernel for this dtype and blocksize, if it does not already exist.
    if cholesky_modules.has_key((dtype, blocksize)):
        mod = cholesky_modules[dtype, blocksize]
    else:
        s = templ_subs(cholesky_template, blocksize=blocksize, dtype=dtype_names[dtype])
        cholesky_sources[dtype, blocksize] = s
        mod = cholesky_modules[dtype, blocksize] = cuda.SourceModule(s)
    
    matrixA_size = numpy.uint32(matrixA_size)
    matrixBlocks = numpy.uint32(matrixA_size/blocksize)
    matrixRest = matrixA_size%blocksize
    
    # if ((matrixRest != 0) and (nx != ny)):
    #    # Matrix is not symmetric or has the wrong dimension -> exit
    #    return None

    if ((matrixA_gpu == None) or (matrixA_size == 0)):
       return None

    cuda_fct_topleft = mod.get_function("d_choldc_topleft")
    cuda_fct_strip = mod.get_function("d_choldc_strip")
    cuda_fct_diagupdate = mod.get_function("d_choldc_diagupdate")
    cuda_fct_loupdate = mod.get_function("d_choldc_loupdate")

    i = (int)(matrixBlocks)
    j = i

    while i > 2:
       logridx = 1
       logridy = i-2
       stripgridx = i-1
       cuda_fct_topleft(matrixA_gpu, matrixA_size, numpy.uint32(j-i), block=(blocksize,blocksize,1), grid=(1,1))
       cuda_fct_strip(matrixA_gpu, matrixA_size, numpy.uint32(j-i), block=(blocksize,blocksize,1), grid=(stripgridx, 1))
       cuda_fct_diagupdate(matrixA_gpu, matrixA_size, numpy.uint32(j-i), block=(blocksize,blocksize,1), grid=(stripgridx, 1))
       cuda_fct_loupdate(matrixA_gpu, matrixA_size, matrixBlocks, numpy.uint32(j-i), block=(blocksize,blocksize,1), grid=(logridx, logridy))
       i = i - 1
    if(j>1):
       cuda_fct_topleft(matrixA_gpu, matrixA_size, numpy.uint32(j-2), block=(blocksize,blocksize,1), grid=(1,1))
       cuda_fct_strip(matrixA_gpu, matrixA_size, numpy.uint32(j-2), block=(blocksize,blocksize,1), grid=(1, 1))
       cuda_fct_diagupdate(matrixA_gpu, matrixA_size, numpy.uint32(j-2), block=(blocksize,blocksize,1), grid=(1, 1))

    cuda_fct_topleft(matrixA_gpu, matrixA_size, numpy.uint32(j-1), block=(blocksize,blocksize,1), grid=(1,1))
    cuda.Context.synchronize()
