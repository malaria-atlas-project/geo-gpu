# Copyright (C) 2009  Bernhard Seiser
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
from numpy import *

def cuda_d_euclidean(x, y, nx, ny, ndx, ndy, cmin, cmax, symm): # Calculate distance matrix for the two input position vectors x and y 

    blocksize = 16

    mod =  cuda.SourceModule("""
    #define BLOCKSIZE 16
    __device__ float d_euclidean(float *x, float *y, int nxi, int nyj, int ndx)
    {
          float d = 0;
	  for(int i = 0; i < ndx; i++)
	  {
		  float dev = x[nxi+i] - y[nyj+i];
                  d += dev*dev;
	  }
          // Test diagonal matrix
          /*
          if(nxi == nyj)
          {
             return 1;
          }
          else
          {
             return 0;
          }
          */
          return sqrt(d);
    }
    __global__ void d_fillMatrix_euclidean_symmetric_full(float *cuda_matrixD, float *x, float *y, int nx, int ndx)
    {
        if(blockIdx.x >= blockIdx.y){
	    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
	    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
	    float d_D_xi_yj = d_euclidean(x,y,nxi,nyj,ndx);
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
            x_gpu = cuda.mem_alloc(nx*ndx*numpy.dtype('float32').itemsize)
            y_gpu = cuda.mem_alloc(nx*ndx*numpy.dtype('float32').itemsize)
            matrixD_gpu = cuda.mem_alloc(nx*nx*numpy.dtype('float32').itemsize)

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

    return None

def cuda_d_matern(matrixC_gpu, nx, ny, cmin, cmax, symm, diff_degree): # Update distance matrix matrixC on GPU 

    blocksize = 16
    nbkmaxstandard = 10

    mod =  cuda.SourceModule("""
    #define NBKMAX 10
    #define BLOCKSIZE 16

    #define xmax_BESS_K    705.342
    #define sqxmin_BESS_K  1.49e-154
    // From http://www.cplusplus.com/reference/clibrary/cfloat/
    #define DBL_MAX 1e+37
    #define DBL_EPSILON 1E-9
    #define DBL_MIN 1e-37

    __const__ __device__ float a = .11593151565841244881;
    __const__ __device__ float m_sqrt_2dpi = .11593151565841244881;
    __const__ __device__ float p[8] = { .805629875690432845,20.4045500205365151,157.705605106676174,536.671116469207504,900.382759291288778,730.923886650660393,229.299301509425145,.822467033424113231 };
    __const__ __device__ float q[7] = { 29.4601986247850434,277.577868510221208,1206.70325591027438,2762.91444159791519,3443.74050506564618,2210.63190113378647,572.267338359892221 };
    __const__ __device__ float r[5] = { -.48672575865218401848,13.079485869097804016,-101.96490580880537526,347.65409106507813131,3.495898124521934782e-4 };
    __const__ __device__ float s[4] = { -25.579105509976461286,212.57260432226544008,-610.69018684944109624,422.69668805777760407 };
    __const__ __device__ float t[6] = { 1.6125990452916363814e-10,2.5051878502858255354e-8,2.7557319615147964774e-6,1.9841269840928373686e-4,.0083333333333334751799,.16666666666666666446 };
    __const__ __device__ float estm[6] = { 52.0583,5.7607,2.7782,14.4303,185.3004, 9.3715 };
    __const__ __device__ float estf[7] = { 41.8341,7.1075,6.4306,42.511,1.35633,84.5096,20.};

    __device__ float ftrunc(float x)
    {
	    if(x >= 0) return floor(x);
	    else return ceil(x);
    }

    __device__ int imin2(int x, int y)
    {
	return (x < y) ? x : y;
    }

    __device__ int imax2(int x, int y)
    {
	return (x < y) ? y : x;
    }

    __device__  void K_bessel(float *x, float *alpha, int *nb, int *ize, float *bk, int *ncalc)
    {
	/* Local variables */
	int iend, i, j, k, m, ii, mplus1;
	float x2by4, twox, c, blpha, ratio, wminf;
	float d1, d2, d3, f0, f1, f2, p0, q0, t1, t2, twonu;
	float dm, ex, bk1, bk2, nu;

	ii = 0; /* -Wall */
	ex = *x;
	nu = *alpha;
	*ncalc = imin2(*nb,0) - 2;
	if (*nb > 0 && (0. <= nu && nu < 1.) && (1 <= *ize && *ize <= 2)) {
	if(ex <= 0 || (*ize == 1 && ex > xmax_BESS_K)) {
	    if(ex <= 0) {
	    if(ex < 0) return;
	    for(i=0; i < *nb; i++)
		bk[i] = (1.0 / 0.0);
	    } else /* would only have underflow */
	    for(i=0; i < *nb; i++)
		bk[i] = 0.;
	    *ncalc = *nb;
	    return;
	}
	k = 0;
	if (nu < sqxmin_BESS_K) {
	    nu = 0.;
	} else if (nu > .5) {
	    k = 1;
	    nu -= 1.;
	}
	twonu = nu + nu;
	iend = *nb + k - 1;
	c = nu * nu;
	d3 = -c;
	if (ex <= 1.) {
	    /* ------------------------------------------------------------
	      Calculation of P0 = GAMMA(1+ALPHA) * (2/X)**ALPHA
		      Q0 = GAMMA(1-ALPHA) * (X/2)**ALPHA
	      ------------------------------------------------------------ */
	    d1 = 0.; d2 = p[0];
	    t1 = 1.; t2 = q[0];
	    for (i = 2; i <= 7; i += 2) {
	    d1 = c * d1 + p[i - 1];
	    d2 = c * d2 + p[i];
	    t1 = c * t1 + q[i - 1];
	    t2 = c * t2 + q[i];
	    }
	    d1 = nu * d1;
	    t1 = nu * t1;
	    f1 = log(ex);
	    f0 = a + nu * (p[7] - nu * (d1 + d2) / (t1 + t2)) - f1;
	    q0 = exp(-nu * (a - nu * (p[7] + nu * (d1-d2) / (t1-t2)) - f1));
	    f1 = nu * f0;
	    p0 = exp(f1);
	    /* -----------------------------------------------------------
	      Calculation of F0 =
	      ----------------------------------------------------------- */
	    d1 = r[4];
	    t1 = 1.;
	    for (i = 0; i < 4; ++i) {
	    d1 = c * d1 + r[i];
	    t1 = c * t1 + s[i];
	    }
	    /* d2 := sinh(f1)/ nu = sinh(f1)/(f1/f0)
	    *       = f0 * sinh(f1)/f1 */
	    if (fabs(f1) <= .5) {
	    f1 *= f1;
	    d2 = 0.;
	    for (i = 0; i < 6; ++i) {
		d2 = f1 * d2 + t[i];
	    }
	    d2 = f0 + f0 * f1 * d2;
	    } else {
	    d2 = sinh(f1) / nu;
	    }
	    f0 = d2 - nu * d1 / (t1 * p0);
	    if (ex <= 1e-10) {
	    /* ---------------------------------------------------------
	      X <= 1.0E-10
	      Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X)
	      --------------------------------------------------------- */
	    bk[0] = f0 + ex * f0;
	    if (*ize == 1) {
		bk[0] -= ex * bk[0];
	    }
	    ratio = p0 / f0;
	    c = ex * DBL_MAX;
	    if (k != 0) {
		/* ---------------------------------------------------
		  Calculation of K(ALPHA,X)
		  and  X*K(ALPHA+1,X)/K(ALPHA,X),    ALPHA >= 1/2
		  --------------------------------------------------- */
		*ncalc = -1;
		if (bk[0] >= c / ratio) {
		return;
		}
		bk[0] = ratio * bk[0] / ex;
		twonu += 2.;
		ratio = twonu;
	    }
	    *ncalc = 1;
	    if (*nb == 1)
		return;

	    /* -----------------------------------------------------
	      Calculate  K(ALPHA+L,X)/K(ALPHA+L-1,X),
	      L = 1, 2, ... , NB-1
	      ----------------------------------------------------- */
	    *ncalc = -1;
	    for (i = 1; i < *nb; ++i) {
		if (ratio >= c)
		return;

		bk[i] = ratio / ex;
		twonu += 2.;
		ratio = twonu;
	    }
	    *ncalc = 1;
	    goto L420;
	    } else {
	    /* ------------------------------------------------------
	      10^-10 < X <= 1.0
	      ------------------------------------------------------ */
	    c = 1.;
	    x2by4 = ex * ex / 4.;
	    p0 = .5 * p0;
	    q0 = .5 * q0;
	    d1 = -1.;
	    d2 = 0.;
	    bk1 = 0.;
	    bk2 = 0.;
	    f1 = f0;
	    f2 = p0;
	    do {
		d1 += 2.;
		d2 += 1.;
		d3 = d1 + d3;
		c = x2by4 * c / d2;
		f0 = (d2 * f0 + p0 + q0) / d3;
		p0 /= d2 - nu;
		q0 /= d2 + nu;
		t1 = c * f0;
		t2 = c * (p0 - d2 * f0);
		bk1 += t1;
		bk2 += t2;
	    } while (fabs(t1 / (f1 + bk1)) > DBL_EPSILON ||
		fabs(t2 / (f2 + bk2)) > DBL_EPSILON);
	    bk1 = f1 + bk1;
	    bk2 = 2. * (f2 + bk2) / ex;
	    if (*ize == 2) {
		d1 = exp(ex);
		bk1 *= d1;
		bk2 *= d1;
	    }
	    wminf = estf[0] * ex + estf[1];
	    }
	} else if (DBL_EPSILON * ex > 1.) {
	    /* -------------------------------------------------
	      X > 1./EPS
	      ------------------------------------------------- */
	    *ncalc = *nb;
	    bk1 = 1. / (m_sqrt_2dpi * sqrt(ex));
	    for (i = 0; i < *nb; ++i) bk[i] = bk1;         
	    return;

	} else {
	    /* -------------------------------------------------------
	      X > 1.0
	      ------------------------------------------------------- */
	    twox = ex + ex;
	    blpha = 0.;
	    ratio = 0.;
	    if (ex <= 4.) {
	    /* ----------------------------------------------------------
	      Calculation of K(ALPHA+1,X)/K(ALPHA,X),  1.0 <= X <= 4.0
	      ----------------------------------------------------------*/
	    d2 = ftrunc(estm[0] / ex + estm[1]);
	    m = (long) d2;
	    d1 = d2 + d2;
	    d2 -= .5;
	    d2 *= d2;
	    for (i = 2; i <= m; ++i) {
		d1 -= 2.;
		d2 -= d1;
		ratio = (d3 + d2) / (twox + d1 - ratio);
	    }
	    /* -----------------------------------------------------------
	      Calculation of I(|ALPHA|,X) and I(|ALPHA|+1,X) by backward
	      recurrence and K(ALPHA,X) from the wronskian
	      -----------------------------------------------------------*/
	    d2 = ftrunc(estm[2] * ex + estm[3]);
	    m = (long) d2;
	    c = fabs(nu);
	    d3 = c + c;
	    d1 = d3 - 1.;
	    f1 = DBL_MIN;
	    f0 = (2. * (c + d2) / ex + .5 * ex / (c + d2 + 1.)) * DBL_MIN;
	    for (i = 3; i <= m; ++i) {
		d2 -= 1.;
		f2 = (d3 + d2 + d2) * f0;
		blpha = (1. + d1 / d2) * (f2 + blpha);
		f2 = f2 / ex + f1;
		f1 = f0;
		f0 = f2;
	    }
	    f1 = (d3 + 2.) * f0 / ex + f1;
	    d1 = 0.;
	    t1 = 1.;
	    for (i = 1; i <= 7; ++i) {
		d1 = c * d1 + p[i - 1];
		t1 = c * t1 + q[i - 1];
	    }
	    p0 = exp(c * (a + c * (p[7] - c * d1 / t1) - log(ex))) / ex;
	    f2 = (c + .5 - ratio) * f1 / ex;
	    bk1 = p0 + (d3 * f0 - f2 + f0 + blpha) / (f2 + f1 + f0) * p0;
	    if (*ize == 1) {
		bk1 *= exp(-ex);
	    }
	    wminf = estf[2] * ex + estf[3];
	    } else {
	    /* ---------------------------------------------------------
	      Calculation of K(ALPHA,X) and K(ALPHA+1,X)/K(ALPHA,X), by
	      backward recurrence, for  X > 4.0
	      ----------------------------------------------------------*/
	    dm = ftrunc(estm[4] / ex + estm[5]);
	    m = (long) dm;
	    d2 = dm - .5;
	    d2 *= d2;
	    d1 = dm + dm;
	    for (i = 2; i <= m; ++i) {
		dm -= 1.;
		d1 -= 2.;
		d2 -= d1;
		ratio = (d3 + d2) / (twox + d1 - ratio);
		blpha = (ratio + ratio * blpha) / dm;
	    }
	    bk1 = 1. / ((m_sqrt_2dpi + m_sqrt_2dpi * blpha) * sqrt(ex));
	    if (*ize == 1)
		bk1 *= exp(-ex);
	    wminf = estf[4] * (ex - fabs(ex - estf[6])) + estf[5];
	    }
	    /* ---------------------------------------------------------
	      Calculation of K(ALPHA+1,X)
	      from K(ALPHA,X) and  K(ALPHA+1,X)/K(ALPHA,X)
	      --------------------------------------------------------- */
	    bk2 = bk1 + bk1 * (nu + .5 - ratio) / ex;
	}
	/*--------------------------------------------------------------------
	  Calculation of 'NCALC', K(ALPHA+I,X),    I  =  0, 1, ... , NCALC-1,
	  &      K(ALPHA+I,X)/K(ALPHA+I-1,X),    I = NCALC, NCALC+1, ... , NB-1
	  -------------------------------------------------------------------*/
	*ncalc = *nb;
	bk[0] = bk1;
	if (iend == 0)
	    return;

	j = 1 - k;
	if (j >= 0)
	    bk[j] = bk2;

	if (iend == 1)
	    return;

	m = imin2((long) (wminf - nu),iend);
	for (i = 2; i <= m; ++i) {
	    t1 = bk1;
	    bk1 = bk2;
	    twonu += 2.;
	    if (ex < 1.) {
	    if (bk1 >= DBL_MAX / twonu * ex)
		break;
	    } else {
	    if (bk1 / ex >= DBL_MAX / twonu)
		break;
	    }
	    bk2 = twonu / ex * bk1 + t1;
	    ii = i;
	    ++j;
	    if (j >= 0) {
	    bk[j] = bk2;
	    }
	}

	m = ii;
	if (m == iend) {
	    return;
	}
	ratio = bk2 / bk1;
	mplus1 = m + 1;
	*ncalc = -1;
	for (i = mplus1; i <= iend; ++i) {
	    twonu += 2.;
	    ratio = twonu / ex + 1./ratio;
	    ++j;
	    if (j >= 1) {
	    bk[j] = ratio;
	    } else {
	    if (bk2 >= DBL_MAX / ratio)
		return;

	    bk2 *= ratio;
	    }
	}
	*ncalc = imax2(1, mplus1 - k);
	if (*ncalc == 1)
	    bk[0] = bk2;
	if (*nb == 1)
	    return;
    L420:
	for (i = *ncalc; i < *nb; ++i) { /* i == *ncalc */
    #ifndef IEEE_754
	    if (bk[i-1] >= DBL_MAX / bk[i])
	    return;
    #endif
	    bk[i] *= bk[i-1];
	    (*ncalc)++;
	}
	}     
    }

    __device__ void d_rkbesl(float x, float alpha, int nb, int ize, float *bk, int ncalc)
    {
          K_bessel(&x, &alpha, &nb, &ize, bk, &ncalc);
    }

    __global__ void d_fillMatrix_rkbesl_symmetric_full(float *cuda_matrixC, float diff_degree, float snu, float rem, int fl, int N, float prefac, int nx)
    {
        if(blockIdx.x >= blockIdx.y){
	    int nxi = blockIdx.x * blockDim.x + threadIdx.x;
	    int nyj = blockIdx.y * blockDim.y + threadIdx.y;
	    float d_C_xi_yj = 1.0;
            float BK[NBKMAX];
            if(cuda_matrixC[nyj*nx + nxi] != 0){
                 d_C_xi_yj = cuda_matrixC[nyj*nx + nxi];
                 d_rkbesl(d_C_xi_yj*snu,rem,fl+1,1,BK,N);
                 d_C_xi_yj = prefac*(pow(d_C_xi_yj,diff_degree))*BK[fl]; //TODO +1??? SEE FORTRAN CODE
            }
	    //__syncthreads;
            cuda_matrixC[nyj*nx + nxi] = d_C_xi_yj;
            cuda_matrixC[nxi*nx + nyj] = d_C_xi_yj;
	}
    }
    """)

    prefac = 1.0
    if diff_degree < 0.0:
       return None
    if diff_degree >= 10.0:
       #TODO gaussian
       return None
    if diff_degree != 1.0:
       ga = numpy.random.gamma(diff_degree)
       prefac = pow(0.5,(diff_degree-1.0)) / ga

    snu = sqrt(diff_degree) * 2.0
    fl = floor(diff_degree) # fl = N
    rem = diff_degree - fl

    if((matrixC_gpu != None) and (symm == True) and (nx == ny)):
       matrixBlocks = nx/blocksize
       if((cmin == 0) and (cmax == nx)):

            #Load cuda function
            cuda_fct = mod.get_function("d_fillMatrix_rkbesl_symmetric_full")

            if(fl + 1 > nbkmaxstandard):
                #TODO: Recompile function with larger NBKMAX and load module
                return None

            #Convert input parameters
            diff_degree = numpy.float32(diff_degree)
            snu = numpy.float32(snu)
            rem = numpy.float32(rem)
            fl = numpy.uint32(fl)
            prefac = numpy.float32(prefac)
            nx = numpy.uint32(nx)

            #Execute cuda function
            cuda_fct(matrixC_gpu, diff_degree, snu, rem, fl, fl, prefac, nx, block=(blocksize,blocksize,1), grid=(matrixBlocks,matrixBlocks))

            #Ouput
            #matrixC_cpu = numpy.ones(nx*nx,numpy.float32)
            #cuda.memcpy_dtoh(matrixC_cpu, matrixC_gpu)
            #print matrixC_cpu

            #return matrixC_gpu
            return matrixC_gpu

    return None

def cuda_d_choleskyDecomposition(matrixA_gpu, matrixA_size): # Choleksy decomposition of matrixA which is already on gpu.

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
    __global__ void d_choldc_topleft(float *m, int matrix_size,  int boffset)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ float topleft[BLOCKSIZE][BLOCKSIZE+1];
        //topleft[ty][tx]=m[ty+BLOCKSIZE*boffset][tx+BLOCKSIZE*boffset];
        topleft[ty][tx]=m[(ty+BLOCKSIZE*boffset)*matrix_size + tx+BLOCKSIZE*boffset];
        __syncthreads();

        float fac;

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

    __global__ void d_choldc_strip(float *m, int matrix_size,  int blockoffset)
    {
        // +1 since blockoffset labels the "topleft" position and boff is the working position...
        int boffx = blockIdx.x+blockoffset+1;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ float topleft[BLOCKSIZE][BLOCKSIZE+1];
        __shared__ float workingmat[BLOCKSIZE][BLOCKSIZE+1];

        topleft[ty][tx] = m[(ty+blockoffset*BLOCKSIZE)*matrix_size +tx+blockoffset*BLOCKSIZE];
        workingmat[tx][ty] = m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];

        //topleft[ty][tx] = m[ty+blockoffset*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];
        //workingmat[tx][ty] = m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

        __syncthreads();

        if(ty==0)
        {
              for (int k=0;k<BLOCKSIZE;k++)
              {
                float dotprod=0.f;
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

    __global__ void d_choldc_diagupdate(float *m, int matrix_size,  int blockoffset)  
    {
        int boffx = blockIdx.x+blockoffset+1;

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ float left[BLOCKSIZE][BLOCKSIZE+1];

        left[ty][tx]=m[(ty+boffx*BLOCKSIZE)*matrix_size + tx+blockoffset*BLOCKSIZE];
        //left[ty][tx]=m[ty+boffx*BLOCKSIZE][tx+blockoffset*BLOCKSIZE];

        __syncthreads();

        float matrixprod=0.f;
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

    __global__ void d_choldc_loupdate(float *m, int matrix_size,  int mat_blocks, int blockoffset)  
    {

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int boffy=blockIdx.y+blockoffset+1;
        int boffx=boffy+1;

        __shared__ float left[BLOCKSIZE][BLOCKSIZE];

        __shared__ float upt[BLOCKSIZE][BLOCKSIZE+1];

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

            float matrixprod=0.f;

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

def maintest(): # EUCLIDEAN, BESSEL, CHOLESKY

    nx = 16 # Matrix dimension
    ny = nx
    ndx = 3 # Position vector dimension (x,y,z) = 3
    ndy = ndx
    x_cpu = numpy.ones(nx*ndx,numpy.float32)
    y_cpu = numpy.zeros(ny*ndy,numpy.float32)

    # Calculate distance matrix on gpu -> returns "pointer" of gpu matrix
    matrixD_gpu = cuda_d_euclidean(x_cpu, y_cpu, nx, ny, ndx, ndy, 0, nx, True)

    # Use distance matrix which has been calculated before as an input for your bessel fct.
    diff_degree = 1.1
    matrixD_gpu = cuda_d_matern(matrixD_gpu, nx, ny, 0, nx, True, diff_degree)

    # Finally carry out the matrix decomposition for the K(D(x_i,y_j)) matrix
    cuda_d_choleskyDecomposition(matrixD_gpu, nx)

    # Copy result from device to host and print it
    matrixK_cpu = numpy.ones(nx*nx,numpy.float32)
    cuda.memcpy_dtoh(matrixK_cpu, matrixD_gpu)
    numpy.set_printoptions(threshold=nan) 
    print matrixK_cpu

if __name__ == "__main__":
    maintest()
    sys.exit()
