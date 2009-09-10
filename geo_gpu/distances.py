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
from jinja2 import Template
from numpy import *

def cuda_d_euclidean(x, y, nx, ny, ndx, ndy, cmin, cmax, symm):

    blocksize = 16

    mod =  cuda.SourceModule("""
    #define BLOCKSIZE 16
    __device__ double d_euclidean(double *x, double *y, int nxi, int nyj, int ndx)
    {
          double d = 0;
	  for(int i = 0; i < ndx; i++)
	  {
		  double dev = x[nxi+i] - y[nyj+i];
                  d += dev*dev;
	  }
          return sqrt(d);
    }
    __global__ void d_fillMatrix_euclidean_symmetric_full(double *cuda_matrixD, double *x, double *y, int nx, int ndx)
    {
        if(blockIdx.x >= blockIdx.y){
	    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
	    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
	    double d_D_xi_yj = d_euclidean(x,y,nxi,nyj,ndx);
	    __syncthreads;
	    cuda_matrixD[nyj*nx + nxi] = d_D_xi_yj;
	    cuda_matrixD[nxi*nx + nyj] = d_D_xi_yj;
	}
    }
    """)

    if((symm == True) and (nx == ny) and (ndx == ndy)):
       matrixBlocks = nx/blocksize
       if((cmin == 0) and (cmax == nx)):

            #Load cuda function
            cuda_fct = mod.get_function("d_fillMatrix_euclidean_symmetric_full")

	    #Allocate arrays on device
            x_gpu = cuda.mem_alloc(nx*ndx*numpy.dtype('float64').itemsize)
            y_gpu = cuda.mem_alloc(nx*ndx*numpy.dtype('float64').itemsize)
            matrixD_gpu = cuda.mem_alloc(nx*nx*numpy.dtype('float64').itemsize)

            #Copy memory from host to device
            cuda.memcpy_htod(x_gpu, x)
            cuda.memcpy_htod(y_gpu, y)

            #Convert input parameters
            nx = numpy.uint32(nx)
            ndx = numpy.uint32(ndx)

            #Execute cuda function
            cuda_fct(matrixD_gpu, x_gpu, y_gpu, nx, ndx, block=(blocksize,blocksize,1), grid=(matrixBlocks,matrixBlocks))

            #Free memory on gpu
            x_gpu.free()
            y_gpu.free()

            #return matrixD_gpu
            return matrixD_gpu
