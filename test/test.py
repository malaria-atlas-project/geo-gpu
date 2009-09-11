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

if __name__ == "__main__":
    nbx = 40
    blocksize=16
    nby = 60
    
    d='float32'
    # d='float'
    x = np.arange(blocksize*nbx,dtype=d)
    y = np.arange(blocksize*nby,dtype=d)    
    
    D = CudaDistance(euclidean, blocksize)
    D.compile_with_parameters()
    
    Ds=D(x,x,symm=True)
    Dns=D(x,y,symm=False)

    C = CudaRawCovariance(exponential, blocksize)
    
    Cs = C(Ds, amp=2., scale=10., symm=True)
    Cns = C(Dns, amp=2., scale=10., symm=False)