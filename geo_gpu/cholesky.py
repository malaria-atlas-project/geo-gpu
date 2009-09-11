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
import sys

# Choleksy decomposition of matrixA which is already on gpu.
def cuda_d_choleskyDecomposition(matrixA_gpu, matrixA_size):

    blocksize = 16
    matrixA_size = numpy.uint32(matrixA_size)
    matrixBlocks = numpy.uint32(matrixA_size/blocksize)
    matrixRest = matrixA_size%blocksize
    
    if ((matrixRest != 0) and (nx != ny)):
       # Matrix is not symmetric or has the wrong dimension -> exit
       return None

    if ((matrixA_gpu == None) or (matrixA_size == 0)):
       return None
 
    mod =  cuda.SourceModule("""
    #define BLOCKSIZE 16
    __global__ void d_choldc_topleft(double *m, int matrix_size,  int boffset)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ double topleft[BLOCKSIZE][BLOCKSIZE+1];
        //topleft[ty][tx]=m[ty+BLOCKSIZE*boffset][tx+BLOCKSIZE*boffset];
        topleft[ty][tx]=m[(ty+BLOCKSIZE*boffset)*matrix_size + tx+BLOCKSIZE*boffset];
        __syncthreads();

        double fac;

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

    __global__ void d_choldc_strip(double *m, int matrix_size,  int blockoffset)
    {
        // +1 since blockoffset labels the "topleft" position and boff is the working position...
        int boffx = blockIdx.x+blockoffset+1;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ double topleft[BLOCKSIZE][BLOCKSIZE+1];
        __shared__ double workingmat[BLOCKSIZE][BLOCKSIZE+1];

        topleft[ty][tx] = m[(ty+blockoffset*BLOCKSIZE)*matrix_size +tx+blockoffset*BLOCKSIZE];
        workingmat[tx][ty] = m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];

        //topleft[ty][tx] = m[ty+blockoffset*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];
        //workingmat[tx][ty] = m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

        __syncthreads();

        if(ty==0)
        {
              for (int k=0;k<BLOCKSIZE;k++)
              {
                double dotprod=0.f;
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

    __global__ void d_choldc_diagupdate(double *m, int matrix_size,  int blockoffset)  
    {
        int boffx = blockIdx.x+blockoffset+1;

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ double left[BLOCKSIZE][BLOCKSIZE+1];

        left[ty][tx]=m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];
        //left[ty][tx]=m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

        __syncthreads();

        double matrixprod=0.f;
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

    __global__ void d_choldc_loupdate(double *m, int matrix_size,  int mat_blocks, int blockoffset)  
    {

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int boffy=blockIdx.y+blockoffset+1;
        int boffx=boffy+1;

        __shared__ double left[BLOCKSIZE][BLOCKSIZE];

        __shared__ double upt[BLOCKSIZE][BLOCKSIZE+1];

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

            double matrixprod=0.f;

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
    """)

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
