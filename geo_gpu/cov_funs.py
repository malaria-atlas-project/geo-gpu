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
from common import *

__all__ = ['CudaRawCovariance','exponential']

class CudaRawCovariance(CudaMatrixFiller):
    """
    Wraps a CUDA covariance function. Compiles it on-the-fly as needed.
    
    - Initialization: c = CudaRawCovariance(exponential, dtype, blocksize, **params)
        Compiles symmetric and unsymmetric versions of the generic kernel
        (CudaRawCovariance.generic) with the given body, dtype, blocksize and
        parameter values. Everything is compiled in as a constant
        
    - Calling: a = c(d,symm=False)
        The appropriate GPU kernel is called and the result is copied to a
        NumPy array and returned. Here 'd' is a matrix of distances.

    - On-GPU calling: a = c.gpu_call(...)
        Just like __call__, but the matrix is left alive on the GPU and is not
        copied into a numpy array. The on-GPU matrix is returned as an opaque
        object.
    """

    generic = """
#define BLOCKSIZE {{blocksize}}

{{preamble}}

__device__ void compute_element__({{dtype}} *d)
{
{{body}}
}
__global__ void compute_matrix__({{dtype}} *cuda_matrix, int nx, int ny)
{
    {{ if symm }}
    if(blockIdx.x >= blockIdx.y){ 
        {{ endif }}
        int nxi = blockIdx.x * blockDim.x + threadIdx.x;
        int nyj = blockIdx.y * blockDim.y + threadIdx.y;
        compute_element__(cuda_matrix + nyj*nx + nxi);
        __syncthreads;
        {{if symm }}
        cuda_matrix[nxi*nx + nyj] = cuda_matrix[nyj*nx + nxi];
    }   {{ endif }}
}"""
        
    def __call__(self,d,symm=False):
        """
        Deallocates all memory used on the GPU, returns result as NumPy matrix.
        """

        c_cpu = np.array(d,dtype=self.dtype,order='F')
        
        nx = c_cpu.shape[0]
        ny = c_cpu.shape[1]
        
        c_gpu = cuda.mem_alloc(nx*ny*self.dtype.itemsize)
        cuda.memcpy_htod(c_gpu, c_cpu)
        self.gpu_call(c_gpu,nx,ny,symm)
        cuda.memcpy_dtoh(c_cpu, c_gpu)
        c_gpu.free()
        
        return c_cpu
        
    def gpu_call(self,c_gpu,nx,ny,symm=False):
        """Leaves the generated matrix on the GPU, returns a PyCuda wrapper."""

        # Compile module if necessary
        mod = self.modules[symm]

        nbx = nx/self.blocksize
        nby = ny/self.blocksize

        #Load cuda function
        cuda_fct = mod.get_function("compute_matrix__")

        #Convert input parameters
        nx = numpy.uint32(nx)
        ny = numpy.uint32(ny)

        #Execute cuda function
        cuda_fct(c_gpu, nx, ny, block=(self.blocksize,self.blocksize,1), grid=(nbx,nby))

        #return matrix_gpu
        return c_gpu

exponential = {'preamble': "", 'params':('amp','scale'),
'body': """
d[0]=exp(-abs(d[0])/{{scale}})*{{amp}}*{{amp}};
"""}

gaussian = {'preamble': "", 'params':('amp','scale'),
'body': """
d[0]=exp(-d[0]*d[0]/{{scale}}/{{scale}})*{{amp}}*{{amp}};
"""}


# TODO: templatize Matern!

# def cuda_d_matern(matrixC_gpu, nx, ny, cmin, cmax, symm, diff_degree):
# 
#     blocksize = 16
#     nbkmaxstandard = 10
# 
#     mod =  cuda.SourceModule(s)
# 
#     prefac = 1.0
#     if diff_degree < 0.0:
#        return None
#     if diff_degree >= 10.0:
#        #TODO gaussian
#        return None
#     if diff_degree != 1.0:
#        ga = numpy.random.gamma(diff_degree)
#        prefac = pow(0.5,(diff_degree-1.0)) / ga
# 
#     snu = sqrt(diff_degree) * 2.0
#     fl = floor(diff_degree) # fl = N
#     rem = diff_degree - fl
# 
#     if((matrixC_gpu != None) and (symm == True) and (nx == ny)):
#        nb = nx/blocksize
#        if((cmin == 0) and (cmax == nx)):
# 
#             #Load cuda function
#             cuda_fct = mod.get_function("d_fillMatrix_rkbesl_symmetric_full")
# 
#             if(fl + 1 > nbkmaxstandard):
#                 #TODO: Recompile function with larger NBKMAX and load module
#                 return None
# 
#             #Convert input parameters
#             diff_degree = numpy.float64(diff_degree)
#             snu = numpy.float64(snu)
#             rem = numpy.float64(rem)
#             fl = numpy.uint32(fl)
#             prefac = numpy.float64(prefac)
#             nx = numpy.uint32(nx)
# 
#             #Execute cuda function
#             cuda_fct(matrixC_gpu, diff_degree, snu, rem, fl, fl, prefac, nx, block=(blocksize,blocksize,1), grid=(nb,nb))
# 
#             #Ouput
#             #matrixC_cpu = numpy.ones(nx*nx,numpy.float64)
#             #cuda.memcpy_dtoh(matrixC_cpu, matrixC_gpu)
#             #print matrixC_cpu
# 
#             #return matrixC_gpu
#             return matrixC_gpu
# 
#     return None

# matern = """
# #define NBKMAX 10
# #define BLOCKSIZE 16
# 
# #define xmax_BESS_K    705.342
# #define sqxmin_BESS_K  1.49e-154
# // From http://www.cplusplus.com/reference/clibrary/cfloat/
# #define DBL_MAX 1e+37
# #define DBL_EPSILON 1E-9
# #define DBL_MIN 1e-37
# 
# __const__ __device__ double a = .11593151565841244881;
# __const__ __device__ double m_sqrt_2dpi = .11593151565841244881;
# __const__ __device__ double p[8] = { .805629875690432845,20.4045500205365151,157.705605106676174,536.671116469207504,900.382759291288778,730.923886650660393,229.299301509425145,.822467033424113231 };
# __const__ __device__ double q[7] = { 29.4601986247850434,277.577868510221208,1206.70325591027438,2762.91444159791519,3443.74050506564618,2210.63190113378647,572.267338359892221 };
# __const__ __device__ double r[5] = { -.48672575865218401848,13.079485869097804016,-101.96490580880537526,347.65409106507813131,3.495898124521934782e-4 };
# __const__ __device__ double s[4] = { -25.579105509976461286,212.57260432226544008,-610.69018684944109624,422.69668805777760407 };
# __const__ __device__ double t[6] = { 1.6125990452916363814e-10,2.5051878502858255354e-8,2.7557319615147964774e-6,1.9841269840928373686e-4,.0083333333333334751799,.16666666666666666446 };
# __const__ __device__ double estm[6] = { 52.0583,5.7607,2.7782,14.4303,185.3004, 9.3715 };
# __const__ __device__ double estf[7] = { 41.8341,7.1075,6.4306,42.511,1.35633,84.5096,20.};
# 
# __device__ double ftrunc(double x)
# {
#     if(x >= 0) return floor(x);
#     else return ceil(x);
# }
# 
# __device__ int imin2(int x, int y)
# {
# return (x < y) ? x : y;
# }
# 
# __device__ int imax2(int x, int y)
# {
# return (x < y) ? y : x;
# }
# 
# __device__  void K_bessel(double *x, double *alpha, int *nb, int *ize, double *bk, int *ncalc)
# {
# /* Local variables */
# int iend, i, j, k, m, ii, mplus1;
# double x2by4, twox, c, blpha, ratio, wminf;
# double d1, d2, d3, f0, f1, f2, p0, q0, t1, t2, twonu;
# double dm, ex, bk1, bk2, nu;
# 
# ii = 0; /* -Wall */
# ex = *x;
# nu = *alpha;
# *ncalc = imin2(*nb,0) - 2;
# if (*nb > 0 && (0. <= nu && nu < 1.) && (1 <= *ize && *ize <= 2)) {
# if(ex <= 0 || (*ize == 1 && ex > xmax_BESS_K)) {
#     if(ex <= 0) {
#     if(ex < 0) return;
#     for(i=0; i < *nb; i++)
#   bk[i] = (1.0 / 0.0);
#     } else /* would only have underflow */
#     for(i=0; i < *nb; i++)
#   bk[i] = 0.;
#     *ncalc = *nb;
#     return;
# }
# k = 0;
# if (nu < sqxmin_BESS_K) {
#     nu = 0.;
# } else if (nu > .5) {
#     k = 1;
#     nu -= 1.;
# }
# twonu = nu + nu;
# iend = *nb + k - 1;
# c = nu * nu;
# d3 = -c;
# if (ex <= 1.) {
#     /* ------------------------------------------------------------
#       Calculation of P0 = GAMMA(1+ALPHA) * (2/X)**ALPHA
#         Q0 = GAMMA(1-ALPHA) * (X/2)**ALPHA
#       ------------------------------------------------------------ */
#     d1 = 0.; d2 = p[0];
#     t1 = 1.; t2 = q[0];
#     for (i = 2; i <= 7; i += 2) {
#     d1 = c * d1 + p[i - 1];
#     d2 = c * d2 + p[i];
#     t1 = c * t1 + q[i - 1];
#     t2 = c * t2 + q[i];
#     }
#     d1 = nu * d1;
#     t1 = nu * t1;
#     f1 = log(ex);
#     f0 = a + nu * (p[7] - nu * (d1 + d2) / (t1 + t2)) - f1;
#     q0 = exp(-nu * (a - nu * (p[7] + nu * (d1-d2) / (t1-t2)) - f1));
#     f1 = nu * f0;
#     p0 = exp(f1);
#     /* -----------------------------------------------------------
#       Calculation of F0 =
#       ----------------------------------------------------------- */
#     d1 = r[4];
#     t1 = 1.;
#     for (i = 0; i < 4; ++i) {
#     d1 = c * d1 + r[i];
#     t1 = c * t1 + s[i];
#     }
#     /* d2 := sinh(f1)/ nu = sinh(f1)/(f1/f0)
#     *       = f0 * sinh(f1)/f1 */
#     if (fabs(f1) <= .5) {
#     f1 *= f1;
#     d2 = 0.;
#     for (i = 0; i < 6; ++i) {
#   d2 = f1 * d2 + t[i];
#     }
#     d2 = f0 + f0 * f1 * d2;
#     } else {
#     d2 = sinh(f1) / nu;
#     }
#     f0 = d2 - nu * d1 / (t1 * p0);
#     if (ex <= 1e-10) {
#     /* ---------------------------------------------------------
#       X <= 1.0E-10
#       Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X)
#       --------------------------------------------------------- */
#     bk[0] = f0 + ex * f0;
#     if (*ize == 1) {
#   bk[0] -= ex * bk[0];
#     }
#     ratio = p0 / f0;
#     c = ex * DBL_MAX;
#     if (k != 0) {
#   /* ---------------------------------------------------
#     Calculation of K(ALPHA,X)
#     and  X*K(ALPHA+1,X)/K(ALPHA,X),    ALPHA >= 1/2
#     --------------------------------------------------- */
#   *ncalc = -1;
#   if (bk[0] >= c / ratio) {
#   return;
#   }
#   bk[0] = ratio * bk[0] / ex;
#   twonu += 2.;
#   ratio = twonu;
#     }
#     *ncalc = 1;
#     if (*nb == 1)
#   return;
# 
#     /* -----------------------------------------------------
#       Calculate  K(ALPHA+L,X)/K(ALPHA+L-1,X),
#       L = 1, 2, ... , NB-1
#       ----------------------------------------------------- */
#     *ncalc = -1;
#     for (i = 1; i < *nb; ++i) {
#   if (ratio >= c)
#   return;
# 
#   bk[i] = ratio / ex;
#   twonu += 2.;
#   ratio = twonu;
#     }
#     *ncalc = 1;
#     goto L420;
#     } else {
#     /* ------------------------------------------------------
#       10^-10 < X <= 1.0
#       ------------------------------------------------------ */
#     c = 1.;
#     x2by4 = ex * ex / 4.;
#     p0 = .5 * p0;
#     q0 = .5 * q0;
#     d1 = -1.;
#     d2 = 0.;
#     bk1 = 0.;
#     bk2 = 0.;
#     f1 = f0;
#     f2 = p0;
#     do {
#   d1 += 2.;
#   d2 += 1.;
#   d3 = d1 + d3;
#   c = x2by4 * c / d2;
#   f0 = (d2 * f0 + p0 + q0) / d3;
#   p0 /= d2 - nu;
#   q0 /= d2 + nu;
#   t1 = c * f0;
#   t2 = c * (p0 - d2 * f0);
#   bk1 += t1;
#   bk2 += t2;
#     } while (fabs(t1 / (f1 + bk1)) > DBL_EPSILON ||
#   fabs(t2 / (f2 + bk2)) > DBL_EPSILON);
#     bk1 = f1 + bk1;
#     bk2 = 2. * (f2 + bk2) / ex;
#     if (*ize == 2) {
#   d1 = exp(ex);
#   bk1 *= d1;
#   bk2 *= d1;
#     }
#     wminf = estf[0] * ex + estf[1];
#     }
# } else if (DBL_EPSILON * ex > 1.) {
#     /* -------------------------------------------------
#       X > 1./EPS
#       ------------------------------------------------- */
#     *ncalc = *nb;
#     bk1 = 1. / (m_sqrt_2dpi * sqrt(ex));
#     for (i = 0; i < *nb; ++i) bk[i] = bk1;         
#     return;
# 
# } else {
#     /* -------------------------------------------------------
#       X > 1.0
#       ------------------------------------------------------- */
#     twox = ex + ex;
#     blpha = 0.;
#     ratio = 0.;
#     if (ex <= 4.) {
#     /* ----------------------------------------------------------
#       Calculation of K(ALPHA+1,X)/K(ALPHA,X),  1.0 <= X <= 4.0
#       ----------------------------------------------------------*/
#     d2 = ftrunc(estm[0] / ex + estm[1]);
#     m = (long) d2;
#     d1 = d2 + d2;
#     d2 -= .5;
#     d2 *= d2;
#     for (i = 2; i <= m; ++i) {
#   d1 -= 2.;
#   d2 -= d1;
#   ratio = (d3 + d2) / (twox + d1 - ratio);
#     }
#     /* -----------------------------------------------------------
#       Calculation of I(|ALPHA|,X) and I(|ALPHA|+1,X) by backward
#       recurrence and K(ALPHA,X) from the wronskian
#       -----------------------------------------------------------*/
#     d2 = ftrunc(estm[2] * ex + estm[3]);
#     m = (long) d2;
#     c = fabs(nu);
#     d3 = c + c;
#     d1 = d3 - 1.;
#     f1 = DBL_MIN;
#     f0 = (2. * (c + d2) / ex + .5 * ex / (c + d2 + 1.)) * DBL_MIN;
#     for (i = 3; i <= m; ++i) {
#   d2 -= 1.;
#   f2 = (d3 + d2 + d2) * f0;
#   blpha = (1. + d1 / d2) * (f2 + blpha);
#   f2 = f2 / ex + f1;
#   f1 = f0;
#   f0 = f2;
#     }
#     f1 = (d3 + 2.) * f0 / ex + f1;
#     d1 = 0.;
#     t1 = 1.;
#     for (i = 1; i <= 7; ++i) {
#   d1 = c * d1 + p[i - 1];
#   t1 = c * t1 + q[i - 1];
#     }
#     p0 = exp(c * (a + c * (p[7] - c * d1 / t1) - log(ex))) / ex;
#     f2 = (c + .5 - ratio) * f1 / ex;
#     bk1 = p0 + (d3 * f0 - f2 + f0 + blpha) / (f2 + f1 + f0) * p0;
#     if (*ize == 1) {
#   bk1 *= exp(-ex);
#     }
#     wminf = estf[2] * ex + estf[3];
#     } else {
#     /* ---------------------------------------------------------
#       Calculation of K(ALPHA,X) and K(ALPHA+1,X)/K(ALPHA,X), by
#       backward recurrence, for  X > 4.0
#       ----------------------------------------------------------*/
#     dm = ftrunc(estm[4] / ex + estm[5]);
#     m = (long) dm;
#     d2 = dm - .5;
#     d2 *= d2;
#     d1 = dm + dm;
#     for (i = 2; i <= m; ++i) {
#   dm -= 1.;
#   d1 -= 2.;
#   d2 -= d1;
#   ratio = (d3 + d2) / (twox + d1 - ratio);
#   blpha = (ratio + ratio * blpha) / dm;
#     }
#     bk1 = 1. / ((m_sqrt_2dpi + m_sqrt_2dpi * blpha) * sqrt(ex));
#     if (*ize == 1)
#   bk1 *= exp(-ex);
#     wminf = estf[4] * (ex - fabs(ex - estf[6])) + estf[5];
#     }
#     /* ---------------------------------------------------------
#       Calculation of K(ALPHA+1,X)
#       from K(ALPHA,X) and  K(ALPHA+1,X)/K(ALPHA,X)
#       --------------------------------------------------------- */
#     bk2 = bk1 + bk1 * (nu + .5 - ratio) / ex;
# }
# /*--------------------------------------------------------------------
#   Calculation of 'NCALC', K(ALPHA+I,X),    I  =  0, 1, ... , NCALC-1,
#   &      K(ALPHA+I,X)/K(ALPHA+I-1,X),    I = NCALC, NCALC+1, ... , NB-1
#   -------------------------------------------------------------------*/
# *ncalc = *nb;
# bk[0] = bk1;
# if (iend == 0)
#     return;
# 
# j = 1 - k;
# if (j >= 0)
#     bk[j] = bk2;
# 
# if (iend == 1)
#     return;
# 
# m = imin2((long) (wminf - nu),iend);
# for (i = 2; i <= m; ++i) {
#     t1 = bk1;
#     bk1 = bk2;
#     twonu += 2.;
#     if (ex < 1.) {
#     if (bk1 >= DBL_MAX / twonu * ex)
#   break;
#     } else {
#     if (bk1 / ex >= DBL_MAX / twonu)
#   break;
#     }
#     bk2 = twonu / ex * bk1 + t1;
#     ii = i;
#     ++j;
#     if (j >= 0) {
#     bk[j] = bk2;
#     }
# }
# 
# m = ii;
# if (m == iend) {
#     return;
# }
# ratio = bk2 / bk1;
# mplus1 = m + 1;
# *ncalc = -1;
# for (i = mplus1; i <= iend; ++i) {
#     twonu += 2.;
#     ratio = twonu / ex + 1./ratio;
#     ++j;
#     if (j >= 1) {
#     bk[j] = ratio;
#     } else {
#     if (bk2 >= DBL_MAX / ratio)
#   return;
# 
#     bk2 *= ratio;
#     }
# }
# *ncalc = imax2(1, mplus1 - k);
# if (*ncalc == 1)
#     bk[0] = bk2;
# if (*nb == 1)
#     return;
# L420:
# for (i = *ncalc; i < *nb; ++i) { /* i == *ncalc */
# #ifndef IEEE_754
#     if (bk[i-1] >= DBL_MAX / bk[i])
#     return;
# #endif
#     bk[i] *= bk[i-1];
#     (*ncalc)++;
# }
# }     
# }
# 
# __device__ void d_rkbesl(double x, double alpha, int nb, int ize, double *bk, int ncalc)
# {
#       K_bessel(&x, &alpha, &nb, &ize, bk, &ncalc);
# }
# 
# __global__ void d_fillMatrix_rkbesl_symmetric_full(double *cuda_matrixC, double diff_degree, double snu, double rem, int fl, int N, double prefac, int nx)
# {
#     if(blockIdx.x >= blockIdx.y){
#     int nxi = blockIdx.x * blockDim.x + threadIdx.x;
#     int nyj = blockIdx.y * blockDim.y + threadIdx.y;
#     double d_C_xi_yj = 1.0;
#         double BK[NBKMAX];
#         if(cuda_matrixC[nyj*nx + nxi] != 0){
#              d_C_xi_yj = cuda_matrixC[nyj*nx + nxi];
#              d_rkbesl(d_C_xi_yj*snu,rem,fl+1,1,BK,N);
#              d_C_xi_yj = prefac*(pow(d_C_xi_yj,diff_degree))*BK[fl]; //TODO +1??? SEE FORTRAN CODE
#         }
#     //__syncthreads;
#         cuda_matrixC[nyj*nx + nxi] = d_C_xi_yj;
#         cuda_matrixC[nxi*nx + nyj] = d_C_xi_yj;
# }
# }
# """