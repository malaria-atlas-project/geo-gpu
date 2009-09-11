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

from geo_gpu import *
import pymc as pm

def maintest():

    ##############################################################################
    # EUCLIDEAN, BESSEL ,CHOLESKY
    ##############################################################################

    nx = 16 #Matrix dimension
    ny = nx
    ndx = 3 #Position vector dimension (x,y,z) = 3
    ndy = ndx
    x_cpu = numpy.ones(nx*ndx,numpy.float64)
    y_cpu = numpy.zeros(ny*ndy,numpy.float64)

    # Calculate distance matrix on gpu -> returns "pointer" of gpu matrix
    matrixD_gpu = cuda_d_euclidean(x_cpu, y_cpu, nx, ny, ndx, ndy, 0, nx, True)

    # Use distance matrix which has been calculated before as an input for your bessel fct.
    diff_degree = 1.1
    matrixD_gpu = cuda_d_matern(matrixD_gpu, nx, ny, 0, nx, True, diff_degree)

    # Finally carry out the matrix decomposition for the K(D(x_i,y_j)) matrix
    cuda_d_choleskyDecomposition(matrixD_gpu, nx)

    # Copy result from device to host and print it
    matrixK_cpu = numpy.ones(nx*nx,numpy.float64)
    cuda.memcpy_dtoh(matrixK_cpu, matrixD_gpu)
    numpy.set_printoptions(threshold=nan) 
    print matrixK_cpu


    sys.exit()


if __name__ == "__main__":
    # maintest()
    nbx = 400
    blocksize=16
    nby = 600
    
    d='float32'
    # d='float'
    x = np.arange(blocksize*nbx,dtype=d)
    y = np.arange(blocksize*nby,dtype=d)    
    d=np.empty((x.shape[0],y.shape[0]),order='F')
    
    c = CudaDistance(euclidean, blocksize)
    c.compile_with_parameters()
    
    a=c(x,x,symm=True)

    import time
    t1=time.time()
    b=c(x,y,symm=False)
    t2=time.time()
    pm.gp.distances.euclidean(d,x,y)
    t3=time.time()
    
    print 'GPU time: %f, CPU time: %f'%(t2-t1,t3-t2)