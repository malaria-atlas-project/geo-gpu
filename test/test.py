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
import geo_gpu
import pymc as pm
import numpy as np
from numpy.testing import assert_almost_equal

def disttest():
    nbx = 40
    blocksize=16
    nby = 60
    
    d='float32'
    # d='float'
    x = np.arange(blocksize*nbx,dtype=d)
    y = np.arange(blocksize*nby,dtype=d)    
    D=np.empty((x.shape[0],y.shape[0]),order='F')
    
    c = CudaDistance(euclidean, blocksize)
    c.compile_with_parameters()
    
    a=c(x,x,symm=True)

    import time
    t1=time.time()
    b=c(x,y,symm=False)
    t2=time.time()
    pm.gp.distances.euclidean(D,x,y)
    t3=time.time()
    
    print 'GPU time: %f, CPU time: %f'%(t2-t1,t3-t2)

def test_correspondence():
    nbx = 1
    blocksize=16
    nby = 2
    
    # nx = blocksize*nbx
    # ny = blocksize*nby
    
    nx = 17
    ny = 17

    d='float32'
    # d='float'
    x = np.arange(nx,dtype=d)
    y = np.arange(ny,dtype=d)    
    # x = np.arange(1,dtype=d)*.25
    # y = np.arange(7,dtype=d)*.25

    D = CudaDistance(euclidean, d, blocksize)
    CR = CudaRawCovariance(matern, d, blocksize, **matern_params(diff_degree = 1.3, amp = 1., scale = 1.))
    
    Cpy = pm.gp.Covariance(pm.gp.matern.euclidean, amp=1., scale=1., diff_degree=1.3)
    
    C = CudaCovariance(D, CR)
    Cnspy = Cpy(x,y)
    Cns = C(x,y)
    
    Cspy = Cpy(y,y)
    Cs = C(y,y)
    
    assert_almost_equal(Cns,Cnspy)
    assert_almost_equal(Cs,Cspy)
    
def test_timing():
    import time

    d='float32'
    
    Cpy = pm.gp.FullRankCovariance(pm.gp.matern.euclidean, amp=1., scale=100., diff_degree=1.3)

    y = pymc.gp.regularize_array(np.arange(2**15,dtype=d)*.25)
    x = y.copy()
    
    for blocksize in [16]:
        print 'Blocksize %i'%blocksize
        # for xsize in np.power(2,[9,10,11,12,13]):            
        for xsize in np.power(2,[10,11,12]):            
            print '\tn=%i'%xsize
            
            ypart = y[:xsize]
            
            D = CudaDistance(euclidean, d, blocksize)
            CR = CudaRawCovariance(matern, d, blocksize, **matern_params(diff_degree = 1.3, amp = 1., scale = 100.))
            C = CudaCovariance(D, CR)
            
            print '\t\tAsymmetric evaluation:'
            t1 = time.time()
            Cnspy = Cpy(x[:xsize],y[:xsize])
            print '\t\t\tFortran: %fs'%(time.time()-t1)
            
            t1 = time.time()
            Cns = C(x[:xsize],y[:xsize])
            print '\t\t\tGPU: %fs'%(time.time()-t1)      
            
            t1 = time.time()
            Cns = C.gpu_call(x[:xsize],y[:xsize])
            print '\t\t\tGPU, no copy back to main: %fs'%(time.time()-t1)      
    
            print '\t\tSymmetric evaluation:'
            t1 = time.time()
            Cspy = Cpy(ypart,ypart)
            print '\t\t\tFortran: %fs'%(time.time()-t1)

            t1 = time.time()
            Cs = C(ypart,ypart)
            print '\t\t\tGPU: %fs'%(time.time()-t1)
            
            t1 = time.time()
            Cs = C.gpu_call(ypart,ypart)
            print '\t\t\tGPU, no copy back to main: %fs'%(time.time()-t1)
            
            print '\t\tSymmetric evaluation and Cholesky:'
            t1 = time.time()  
            Spy = Cpy.cholesky(ypart)
            print '\t\t\tFortran: %fs'%(time.time()-t1)
            
            t1 = time.time()
            S = C.cholesky(ypart)
            print '\t\t\tGPU: %fs'%(time.time()-t1)
            
            t1 = time.time()
            S = C.gpu_cholesky(ypart, blocksize=blocksize)
            print '\t\t\tGPU, no copy back to main: %fs'%(time.time()-t1)
    
if __name__ == "__main__":
    # test_timing()
    test_correspondence()