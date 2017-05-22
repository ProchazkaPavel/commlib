""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cimport cython
import ctypes

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef inline double float_max(double a, double b): return a if a >= b else b
cdef inline long int_max(long a, long b): return a if a >= b else b

#cdef inline double float_max4(double a1, double a2, double a3, double a4):
#     return float_max(float_max(a1,a2), float_max(a3,a4))
cdef inline double float_max4(double a1, double a2, double a3, double a4):\
     return float_max(a1, a2) if (float_max(a1, a2) >= float_max(a3, a4)) else float_max(a3, a4)
cdef inline long int_max4(long a1, long a2, long a3, long a4):\
     return int_max(a1, a2) if (int_max(a1, a2) >= int_max(a3, a4)) else int_max(a3, a4)

cdef inline double float_max8(double a1, double a2, double a3, double a4,double a5, double a6, double a7, double a8):
     return float_max(float_max(float_max(a1,a2), float_max(a3,a4)), float_max(float_max(a5,a6), float_max(a7,a8)))

cdef extern from "Turbo_lib.h":
  void update_FSM(int N, double *pc, double *pd, double *out)
  void update_FSM_GF4(int N, double *pcv, double *pdv, double *out)
  void update_FSM_GFq(int N, double *pcv, double *pdv, double *out, int q)
  void update_general_FSM(int N, double *pcv, double *pdv, double *out, long Md, long Ms, long Mc, long *S, long *Q)
  void update_general_FSM_both(int N, double *pcv, double *pdv, double *outc, double *outd, long Md, long Ms, long Mc, long *S, long *Q)
  void eff_turbo_update(unsigned int N, float *b, float *c1, float *c2, unsigned int Md, unsigned int Ms, unsigned int Mc, unsigned int *S, unsigned int *Q, unsigned int *inter, unsigned int *deinter, unsigned int K)


# create the wrapper code, with numpy type annotations
def update_general_FSM_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
		   np.ndarray[double, ndim=1, mode="c"] pd not None,
		   np.ndarray[double, ndim=1, mode="c"] out not None,
		   np.ndarray[long, ndim=2, mode="c"] S not None,
		   np.ndarray[long, ndim=2, mode="c"] Q not None):
    update_general_FSM  (pd.shape[0]/Q.shape[0],<double*> np.PyArray_DATA(pc),
		<double*> np.PyArray_DATA(pd),
		<double*> np.PyArray_DATA(out), 
		Q.shape[0], Q.shape[1], np.max(Q) + 1, 
		<long*> np.PyArray_DATA(S.flatten(1)), 
		<long*> np.PyArray_DATA(Q.flatten(1)))

def update_general_FSM_both_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
		   np.ndarray[double, ndim=1, mode="c"] pd not None,
		   np.ndarray[double, ndim=1, mode="c"] outc not None,
		   np.ndarray[double, ndim=1, mode="c"] outd not None,
		   np.ndarray[long, ndim=2, mode="c"] S not None,
		   np.ndarray[long, ndim=2, mode="c"] Q not None):
    update_general_FSM_both  (pd.shape[0]/Q.shape[0],<double*> np.PyArray_DATA(pc),
		<double*> np.PyArray_DATA(pd),
		<double*> np.PyArray_DATA(outc), 
		<double*> np.PyArray_DATA(outd), 
		Q.shape[0], Q.shape[1], np.max(Q) + 1, 
		<long*> np.PyArray_DATA(S.flatten(1)), 
		<long*> np.PyArray_DATA(Q.flatten(1)))

def update_FSM_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
		   np.ndarray[double, ndim=1, mode="c"] pd not None,
		   np.ndarray[double, ndim=1, mode="c"] out not None):
    update_FSM  (pc.shape[0],<double*> np.PyArray_DATA(pc),
		<double*> np.PyArray_DATA(pd),
		<double*> np.PyArray_DATA(out))

def update_FSM_GF4_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
		   np.ndarray[double, ndim=1, mode="c"] pd not None,
		   np.ndarray[double, ndim=1, mode="c"] out not None):
    update_FSM_GF4  (pc.shape[0]/4,<double*> np.PyArray_DATA(pc),
		<double*> np.PyArray_DATA(pd),
		<double*> np.PyArray_DATA(out))

def update_FSM_GFq_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
		   np.ndarray[double, ndim=1, mode="c"] pd not None,
		   np.ndarray[double, ndim=1, mode="c"] out not None, q):
    update_FSM_GFq  (pc.shape[0]/q,<double*> np.PyArray_DATA(pc),
		<double*> np.PyArray_DATA(pd),
		<double*> np.PyArray_DATA(out), q)


def update_general_FSM_C(np.ndarray[double, ndim=2, mode="c"] pc not None,
                         np.ndarray[double, ndim=2, mode="c"] pd not None,
                         np.ndarray[double, ndim=2, mode="c"] out not None,
                         np.ndarray[unsigned int, ndim=2, mode="c"] S not None, 
                         np.ndarray[unsigned int, ndim=2, mode="c"] Q not None, 
                         unsigned int Md, unsigned int Ms, unsigned int Mc, unsigned int N): 
  cdef unsigned int i;
  cdef unsigned int j;
  cdef unsigned int k; 
  cdef unsigned int index;
  cdef double m
  
  cdef np.ndarray[double, ndim=2] lf = np.zeros([N + 1, Ms], dtype=float)
  cdef np.ndarray[double, ndim=2] lb = np.zeros([N + 1, Ms], dtype=float)
  lf[0,0] = 1;
  for i in range(1, Ms):
    lf[0,i] = 0 # 0 zero initial state

  for i in range(Ms):
    lb[N,i] = 1./Ms # unknown final state

  for i in range(N): # Forward
    for j in range(Ms):
      for k in range(Md):
        lf[i + 1, S[k, j]] += lf[i, j] * pd[i,k]*pc[i,Q[k,j]]
    m = 0
    for j in range(Ms):
      m += lf[i + 1, j]
    m = 1./m
    for j in range(Ms):
      lf[i + 1, j] = lf[i + 1, j] * m

#    lf[i + 1, :] = lf[i + 1, :] / (lf[i + 1, :]).sum()  # Norming

  for index in range(N): # Backward
    for j in range(Ms):
      for k in range(Md):
        i = N - 1 - index
        lb[i, j] += lb[i + 1, S[k, j]] * pc[i ,Q[k, j]] * pd[i, k]
    m = 0
    for j in range(Ms):
      m += lb[i, j]
    m = 1./m
    for j in range(Ms):
      lb[i, j] = lb[i, j] * m

  for i in range(N): # To data
    for j in range(Ms):
      for k in range(Md):
        out[i, k] += lb[i+1, S[k, j]] * pc[i, Q[k, j]] * lf[i, j]      
    m = 0
    for k in range(Md):
      m += out[i, k]
    m = 1./m
    for k in range(Md):
      out[i, k] = out[i, k] * m
#    out[i, :] = out[i, :] / (out[i, :]).sum()  # Norming



def stream_turbo_decode_funct(np.ndarray[float] b not None,
                              np.ndarray[float] c1 not None,
                              np.ndarray[float] c2 not None,
                              np.ndarray[unsigned int] S not None,
                              np.ndarray[unsigned int] Q not None,
                              np.ndarray[unsigned int] inter not None,
                              np.ndarray[unsigned int] deinter not None, K):       
    cdef int m, n

    # declare a numpy array of raw bytes (unsigned 8-bit integers)
    # and assign it to a view of the input data.
    cdef np.uint8_t[:] buffb
    buffb = b.view(np.uint8)

    cdef np.uint8_t[:] buffc1
    buffc1 = c1.view(np.uint8)

    cdef np.uint8_t[:] buffc2
    buffc2 = c2.view(np.uint8)

    cdef np.uint8_t[:] buffS
    buffS = S.view(np.uint8)

    cdef np.uint8_t[:] buffQ
    buffQ = Q.view(np.uint8)

    cdef np.uint8_t[:] buff_inter
    buff_inter = inter.view(np.uint8)

    cdef np.uint8_t[:] buff_deinter
    buff_deinter = deinter.view(np.uint8)


    N = np.uint32(len(inter))
    Md = np.uint32(len(b)/len(inter))
    Ms = np.uint32(len(S) / Md)
    Mc = np.uint32(len(c1)/len(inter))
    eff_turbo_update(N, 
                     <float *>&buffb[0],
                     <float *>&buffc1[0], 
                     <float *>&buffc2[0], 
                     Md, Ms, Mc, 
                     <unsigned int *>&buffS[0], 
                     <unsigned int *>&buffQ[0],  
                     <unsigned int *>&buff_inter[0],  
                     <unsigned int *>&buff_deinter[0], 
                     np.uint32(K))


@cython.boundscheck(False) # turn of bounds-checking for entire function
def create_matrix_SQ_MS(np.ndarray[long, ndim=2, mode="c"] S not None,
                   np.ndarray[long, ndim=2, mode="c"] Q not None):
    cdef unsigned int Md, Ms 
    Md, Ms = np.shape(S) 
    cdef unsigned int s, d
    cdef unsigned int s0, s1, s2
    cdef unsigned int Nb = <unsigned int>np.log2(Md)
    cdef unsigned int N2, Np2
    N2 = 2*Nb
    Md2 = Md**2
    mask = ((1 << Nb) - 1)
    for s in range(Ms):
      for d in range(Md):
          s0 = (s >> N2)
          s1 = ((s >> Nb) & mask)
          s2 = (s & mask)
          S[d,s] = Md2 * ((s2 + d) % Md) + Md * ((s0 + s2) % Md) + s1
          Q[d,s] = (s2 + s0 + d) % Md
     
@cython.boundscheck(False) # turn of bounds-checking for entire function
def create_matrix_SQ_XOR(np.ndarray[long, ndim=2, mode="c"] S not None,
                   np.ndarray[long, ndim=2, mode="c"] Q not None):
    cdef unsigned int Md, Ms 
    Md, Ms = np.shape(S) 
    cdef unsigned int s, d
    cdef unsigned int s0, s1, s2
    cdef unsigned int Nb = <unsigned int>np.log2(Md)
    cdef unsigned int N2, Np2
    N2 = 2*Nb
    Md2 = Md**2
    mask = ((1 << Nb) - 1)
    for s in range(Ms):
      for d in range(Md):
          s0 = (s >> N2)
          s1 = ((s >> Nb) & mask)
          s2 = (s & mask)
          S[d,s] = Md2 * ((s2 ^ d)) + Md * ((s0 ^ s2)) + s1
          Q[d,s] = (s2 ^ s0 ^ d) 
     
@cython.boundscheck(False) # turn of bounds-checking for entire function
def create_matrix_SQ_XOR_mod(np.ndarray[long, ndim=2, mode="c"] S not None,
                   np.ndarray[long, ndim=2, mode="c"] Q not None):
    cdef unsigned int Md, Ms 
    Md, Ms = np.shape(S) 
    cdef unsigned int s, d
    cdef unsigned int s0, s1, s2
    cdef unsigned int Nb = <unsigned int>np.log2(Md)
    cdef unsigned int N2, Np2
    N2 = 2*Nb
    Md2 = Md**2
    mask = ((1 << Nb) - 1)
    for s in range(Ms):
      for d in range(Md):
          s2 = (s >> N2)
          s1 = ((s >> Nb) & mask)
          s0 = (s & mask)
          S[d,s] = ((s2 ^ d)) + Md * ((s0 ^ s2)) + s1 * Md2
          Q[d,s] = (s2 ^ s0 ^ d) 
     
@cython.boundscheck(False) # turn of bounds-checking for entire function
def create_matrix_SQ_MS_mod(np.ndarray[long, ndim=2, mode="c"] S not None,
                   np.ndarray[long, ndim=2, mode="c"] Q not None):
    cdef unsigned int Md, Ms 
    Md, Ms = np.shape(S) 
    cdef unsigned int s, d
    cdef unsigned int s0, s1, s2
    cdef unsigned int Nb = <unsigned int>np.log2(Md)
    cdef unsigned int N2, Np2
    N2 = 2*Nb
    Md2 = Md**2
    mask = ((1 << Nb) - 1)
    for s in range(Ms):
      for d in range(Md):
          s2 = (s >> N2)
          s1 = ((s >> Nb) & mask)
          s0 = (s & mask)
          S[d,s] = ((s2 + d)%Md) + Md * ((s0 + s2)%Md) + s1 * Md2
          Q[d,s] = (s2 + s0 + d) % Md
     

@cython.boundscheck(False) # turn of bounds-checking for entire function
def create_matrix_SQ_XOR_all(np.ndarray[long, ndim=2, mode="c"] S not None, # s_n+1 = S(dn, sn)
                   np.ndarray[long, ndim=2, mode="c"] Q not None,    # cn = Q(dn, sn)
                   np.ndarray[long, ndim=2, mode="c"] S1 not None,   # sn = S1(cn, s_n+1)
                   np.ndarray[long, ndim=2, mode="c"] Q1 not None,   # dn = Q1(cn, s_n+1)
                   np.ndarray[long, ndim=2, mode="c"] Qm1 not None): # dn = Qm1(cn, sn)
    cdef unsigned int Md, Ms 
    Md, Ms = np.shape(S) 
    cdef unsigned int s, d
    cdef unsigned int s0, s1, s2
    cdef unsigned int Nb = <unsigned int>np.log2(Md)
    cdef unsigned int N2, Np2
    N2 = 2*Nb
    Md2 = Md**2
    mask = ((1 << Nb) - 1)
    for s in range(Ms):
      for d in range(Md):
          s0 = (s >> N2)
          s1 = ((s >> Nb) & mask)
          s2 = (s & mask)
          S[d,s] = Md2 * ((s2 ^ d)) + Md * ((s0 ^ s2)) + s1
          Q[d,s] = (s2 ^ s0 ^ d) 
          S1[Q[d,s], S[d,s]] = s
          Q1[Q[d,s], S[d,s]] = d
          Qm1[Q[d,s], s] = d
     
@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_encode(np.ndarray[long, ndim=1, mode="c"] d not None,
               np.ndarray[long, ndim=1, mode="c"] c not None,
               np.ndarray[long, ndim=2, mode="c"] S not None,
               np.ndarray[long, ndim=2, mode="c"] Q not None):
    cdef unsigned int N = len(d)
    cdef unsigned int i
    cdef unsigned int s = 0
    for i in range(N):
      c[i] = Q[d[i], s]
      s = S[d[i], s]




#### LLR-based FSM updates functions
# Update for S = Q = [[0,1],[1,0]] (K=1)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def update_FSM_LLR1(np.ndarray[double, ndim=1, mode="c"] lc not None,
                   np.ndarray[double, ndim=1, mode="c"] ld not None,
                   np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef unsigned int N = len(lc) 
    cdef np.ndarray[double, ndim=1] lf = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb = np.zeros(N + 1, dtype=float)
    cdef double num
    cdef double denum
    cdef unsigned int i, ind
    lf[0] = -200 # zero initial state
    for i in range(N):
      num = float_max(lf[i]+lc[i], ld[i]+lc[i])
      denum = float_max(lf[i]+ld[i], 0)
      lf[i+1] = num-denum

#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N):
#      i = N - 1- ind
      i =  <unsigned int>(N - 1- ind)
      num = float_max(lb[i+1]+lc[i], ld[i])
      denum = float_max(0, ld[i]+lb[i+1]+lc[i])
#      print i, num, denum
      lb[i] = num-denum
    
    for i in range(N):
      num = float_max(lb[1+i]+lc[i], lf[i])
      denum = float_max(lf[i]+lc[i]+lb[i+1], 0)
      out[i] = num-denum

@cython.boundscheck(False) # turn of bounds-checking for entire function
def update_FSM_LLR1_int(np.ndarray[unsigned int, ndim=1, mode="c"] lc not None,
                   np.ndarray[unsigned int, ndim=1, mode="c"] ld not None,
                   np.ndarray[unsigned int, ndim=1, mode="c"] out not None):
    cdef unsigned int N = len(lc) 
    cdef unsigned int z = (1 << 30)
    cdef unsigned int z2 = (1 << 31)
    cdef np.ndarray[unsigned int, ndim=1] lf = np.zeros(N + 1, dtype=np.uint32)
    cdef np.ndarray[unsigned int, ndim=1] lb = z * np.ones(N + 1, dtype=np.uint32)
    cdef unsigned int num
    cdef unsigned int denum
    cdef unsigned int i, ind
    for i in range(N):
      num = int_max(<unsigned int>(lf[i] + lc[i] - z), <unsigned int>(ld[i]+lc[i]-z))
      denum = int_max(<unsigned int>(lf[i] + ld[i] - z), z)
      lf[i+1] = <unsigned int>(z+num-denum)

#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N):
#      i = N - 1- ind
      i =  <unsigned int>(N - 1- ind)
      num = <unsigned int>int_max(<unsigned int>(lb[i+1]+lc[i]-z), <unsigned int>ld[i])
      denum = <unsigned int>int_max(z, <unsigned int>(ld[i]+lb[i+1]+lc[i]-z2))
#      print i, num, denum
      lb[i] = <unsigned int>(z+num-denum)
    
    for i in range(N):
      num = <unsigned int>(int_max(<unsigned int>(lb[1+i]+lc[i]-z), <unsigned int>(lf[i])))
      denum = <unsigned int>(int_max(<unsigned int>(lf[i]+lc[i]+lb[i+1]-z2), z))
      out[i] = <unsigned int>(z + num -denum)

# Update for S = [[0, 2, 1, 3],[1, 3, 0, 2]], Q=[[0, 0, 1, 1],[1, 1, 0, 0]] (K=2)
@cython.boundscheck(False) # turn of bounds-checking for entire function
def update_FSM_LLR2(np.ndarray[double, ndim=1, mode="c"] lc not None,
                   np.ndarray[double, ndim=1, mode="c"] ld not None,
                   np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef unsigned int N = len(lc) 
    cdef np.ndarray[double, ndim=1] lf1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lf2 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb2 = np.zeros(N + 1, dtype=float)
    cdef double num
    cdef double denum
    cdef unsigned int i, ind
    lf1[0] = -200 # zero initial state
    lf2[0] = -200 # zero initial state
    for i in range(N):
      num = float_max4(lf2[i], lf1[i] + lf2[i] + lc[i], ld[i] + lf2[i] + lc[i], ld[i] + lf1[i] + lf2[i])
      denum = float_max4(0, lf1[i] + lc[i], ld[i] + lc[i], ld[i] + lf1[i])
      lf1[i+1] = num-denum
      num = float_max4(lf1[i] + lc[i], lf1[i] + lf2[i] + lc[i], ld[i] + lc[i], ld[i] + lf2[i] + lc[i]) 
      denum = float_max4(0, lf2[i], ld[i] + lf1[i], ld[i] + lf1[i] + lf2[i])
      lf2[i+1] = num-denum

#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N):
      i =  <unsigned int>(N - 1- ind)
      num = float_max4(lb2[i+1] + lc[i], lb1[i+1] + lb2[i+1] + lc[i], ld[i], ld[i] + lb1[i+1])                        
      denum = float_max4(0 ,lb1[i+1], ld[i]+lb2[i+1]+lc[i],ld[i]+lb1[i+1]+lb2[i+1]+lc[i])
      lb1[i] = num - denum
      num = float_max4(lb1[i+1], lb1[i+1]+lb2[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1]+lc[i], ld[i]+lb1[i+1])
      denum = float_max4(0,lb2[i+1]+lc[i], ld[i]+lb2[i+1]+lc[i], ld[i])
      lb2[i] = num - denum
    
    for i in range(N):
      num = float_max4(lb2[i+1] + lc[i], lf2[i] + lb1[i+1] + lb2[i+1] + lc[i], lf1[i], lf1[i] + lf2[i] + lb1[i+1])
      denum = float_max4(0, lf2[i] + lb1[i+1], lf1[i] + lb2[i+1] + lc[i], lf1[i] + lf2[i] + lb1[i+1] + lb2[i+1] + lc[i])
      out[i] = num-denum

# Update for S = [[0, 2, 1, 3, 6, 4, 7, 5], [4, 6, 5, 7, 2, 0, 3, 1]]
#            Q = [[0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1]]
#            K = 3
@cython.boundscheck(False) # turn of bounds-checking for entire function
def update_FSM_LLR3(np.ndarray[double, ndim=1, mode="c"] lc not None,
                   np.ndarray[double, ndim=1, mode="c"] ld not None,
                   np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef unsigned int N = len(lc) 
    cdef np.ndarray[double, ndim=1] lf1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lf2 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lf3 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb2 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb3 = np.zeros(N + 1, dtype=float)
    cdef double num
    cdef double denum
    cdef unsigned int i, ind
    lf1[0] = -200 # zero initial state
    lf2[0] = -200 # zero initial state
    lf3[0] = -200 # zero initial state
    for i in range(N): # forward
#      num =  float_max8(lf1[i]+lb2[i+1]+lc[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb3[i+1], ld[i]+lc[i], ld[i]+lf3[i]+lb2[i+1], ld[i]+lf2[i]+lb3[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      num =  float_max8(lf1[i]+lc[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lc[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lc[i], ld[i]+lf3[i], ld[i]+lf2[i]+lc[i], ld[i]+lf2[i]+lf3[i])
      denum =  float_max8(0 , lf3[i]+lc[i], lf2[i], lf2[i]+lf3[i]+lc[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      lf1[i+1]  = num - denum

#      num =  float_max8(lf3[i]+lc[i], lf2[i]+lf3[i]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lf1[i], ld[i]+lf1[i]+lf2[i]+lb3[i+1])
#      denum =  float_max8(0 , lf2[i]+lb3[i+1], lf1[i]+lf3[i]+lb1[i+1], lf1[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      num =  float_max8(lf3[i]+lc[i], lf2[i]+lf3[i]+lc[i], lf1[i]+lc[i], lf1[i]+lf2[i]+lc[i], ld[i]+lf3[i], ld[i]+lf2[i]+lf3[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf2[i])
      denum =  float_max8(0 , lf2[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lc[i], ld[i]+lf2[i]+lc[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      lf2[i+1]  = num - denum

#      num =  float_max8(lf2[i], lf2[i]+lf3[i]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb1[i+1], ld[i]+lf2[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i])
      num =  float_max8(lf2[i], lf2[i]+lf3[i]+lc[i], lf1[i]+lf2[i]+lc[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lf2[i]+lc[i], ld[i]+lf2[i]+lf3[i], ld[i]+lf1[i]+lf2[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      denum =  float_max8(0 , lf3[i]+lc[i], lf1[i]+lc[i], lf1[i]+lf3[i], ld[i]+lc[i], ld[i]+lf3[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf3[i]+lc[i])
      lf3[i+1]  = num - denum
    
#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N): # backward
      i =  <unsigned int>(N - 1- ind)
#      num =  float_max8(lb1[i+1]+lb2[i+1]+lc[i], lf3[i]+lb1[i+1], lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb2[i+1], ld[i]+lf3[i]+lc[i], ld[i]+lf2[i]+lb2[i+1]+lb3[i+1], ld[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1])
      num =  float_max8(lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb3[i+1], ld[i]+lb2[i+1], ld[i]+lc[i], ld[i]+lb2[i+1]+lb3[i+1], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb2[i+1]+lc[i], lb3[i+1], lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1])
      lb1[i]  = num - denum

#      num =  float_max8(lb3[i+1], lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lf3[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i])
      num =  float_max8(lb3[i+1], lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lb2[i+1]+lb3[i+1], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb2[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb2[i+1], ld[i]+lc[i])
      lb2[i]  = num - denum

#      num =  float_max8(lb2[i+1]+lc[i], lf2[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1], lf1[i]+lf2[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf2[i]+lb3[i+1], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1])
      num =  float_max8(lb2[i+1]+lc[i], lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1], lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lc[i], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb3[i+1], lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb2[i+1], ld[i]+lb2[i+1]+lb3[i+1])
      lb3[i]  = num - denum

    
    for i in range(N): #data
      num =  float_max8(lb1[i+1]+lc[i], lf3[i]+lb1[i+1]+lb2[i+1], lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], lf1[i]+lb2[i+1], lf1[i]+lf3[i]+lc[i], lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1], lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1])
      out[i] = num-denum

@cython.boundscheck(False) # turn of bounds-checking for entire function
def update_FSM_LLR3_both(np.ndarray[double, ndim=1, mode="c"] lc not None,
                   np.ndarray[double, ndim=1, mode="c"] ld not None,
                   np.ndarray[double, ndim=1, mode="c"] outc not None,
                   np.ndarray[double, ndim=1, mode="c"] outd not None):
    cdef unsigned int N = len(lc) 
    cdef np.ndarray[double, ndim=1] lf1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lf2 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lf3 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb1 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb2 = np.zeros(N + 1, dtype=float)
    cdef np.ndarray[double, ndim=1] lb3 = np.zeros(N + 1, dtype=float)
    cdef double num
    cdef double denum
    cdef unsigned int i, ind
    lf1[0] = -200 # zero initial state
    lf2[0] = -200 # zero initial state
    lf3[0] = -200 # zero initial state
    for i in range(N): # forward
#      num =  float_max8(lf1[i]+lb2[i+1]+lc[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb3[i+1], ld[i]+lc[i], ld[i]+lf3[i]+lb2[i+1], ld[i]+lf2[i]+lb3[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      num =  float_max8(lf1[i]+lc[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lc[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lc[i], ld[i]+lf3[i], ld[i]+lf2[i]+lc[i], ld[i]+lf2[i]+lf3[i])
      denum =  float_max8(0 , lf3[i]+lc[i], lf2[i], lf2[i]+lf3[i]+lc[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      lf1[i+1]  = num - denum

#      num =  float_max8(lf3[i]+lc[i], lf2[i]+lf3[i]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lf1[i], ld[i]+lf1[i]+lf2[i]+lb3[i+1])
#      denum =  float_max8(0 , lf2[i]+lb3[i+1], lf1[i]+lf3[i]+lb1[i+1], lf1[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      num =  float_max8(lf3[i]+lc[i], lf2[i]+lf3[i]+lc[i], lf1[i]+lc[i], lf1[i]+lf2[i]+lc[i], ld[i]+lf3[i], ld[i]+lf2[i]+lf3[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf2[i])
      denum =  float_max8(0 , lf2[i], lf1[i]+lf3[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lc[i], ld[i]+lf2[i]+lc[i], ld[i]+lf1[i]+lf3[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      lf2[i+1]  = num - denum

#      num =  float_max8(lf2[i], lf2[i]+lf3[i]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb1[i+1], ld[i]+lf2[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i])
      num =  float_max8(lf2[i], lf2[i]+lf3[i]+lc[i], lf1[i]+lf2[i]+lc[i], lf1[i]+lf2[i]+lf3[i], ld[i]+lf2[i]+lc[i], ld[i]+lf2[i]+lf3[i], ld[i]+lf1[i]+lf2[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lc[i])
      denum =  float_max8(0 , lf3[i]+lc[i], lf1[i]+lc[i], lf1[i]+lf3[i], ld[i]+lc[i], ld[i]+lf3[i], ld[i]+lf1[i], ld[i]+lf1[i]+lf3[i]+lc[i])
      lf3[i+1]  = num - denum
    
#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N): # backward
      i =  <unsigned int>(N - 1- ind)
#      num =  float_max8(lb1[i+1]+lb2[i+1]+lc[i], lf3[i]+lb1[i+1], lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb2[i+1], ld[i]+lf3[i]+lc[i], ld[i]+lf2[i]+lb2[i+1]+lb3[i+1], ld[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1])
      num =  float_max8(lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb3[i+1], ld[i]+lb2[i+1], ld[i]+lc[i], ld[i]+lb2[i+1]+lb3[i+1], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb2[i+1]+lc[i], lb3[i+1], lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1])
      lb1[i]  = num - denum

#      num =  float_max8(lb3[i+1], lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lf3[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf3[i]+lc[i])
      num =  float_max8(lb3[i+1], lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lb2[i+1]+lb3[i+1], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb2[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb2[i+1], ld[i]+lc[i])
      lb2[i]  = num - denum

#      num =  float_max8(lb2[i+1]+lc[i], lf2[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1], lf1[i]+lf2[i]+lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lc[i], ld[i]+lf1[i]+lf2[i]+lb3[i+1]+lc[i])
#      denum =  float_max8(0 , lf2[i]+lb3[i+1], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1])
      num =  float_max8(lb2[i+1]+lc[i], lb2[i+1]+lb3[i+1]+lc[i], lb1[i+1], lb1[i+1]+lb3[i+1], ld[i]+lb1[i+1]+lb2[i+1], ld[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lc[i], ld[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lb3[i+1], lb1[i+1]+lb2[i+1]+lc[i], lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], ld[i]+lb1[i+1]+lc[i], ld[i]+lb1[i+1]+lb3[i+1]+lc[i], ld[i]+lb2[i+1], ld[i]+lb2[i+1]+lb3[i+1])
      lb3[i]  = num - denum

    
    for i in range(N): #data
      num =  float_max8(lb1[i+1]+lc[i], lf3[i]+lb1[i+1]+lb2[i+1], lf2[i]+lb1[i+1]+lb3[i+1]+lc[i], lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], lf1[i]+lb2[i+1], lf1[i]+lf3[i]+lc[i], lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1], lf1[i]+lf2[i]+lf3[i]+lb3[i+1]+lc[i])
      denum =  float_max8(0 , lf3[i]+lb2[i+1]+lc[i], lf2[i]+lb3[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lb1[i+1]+lb2[i+1]+lc[i], lf1[i]+lf3[i]+lb1[i+1], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1]+lc[i], lf1[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1])
      outd[i] = num-denum

    for i in range(N): #Codeword
      num =  float_max8(lf3[i]+lb2[i+1], lf2[i]+lf3[i]+lb2[i+1]+lb3[i+1], lf1[i]+lb1[i+1]+lb2[i+1], lf1[i]+lf2[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lb1[i+1], ld[i]+lf2[i]+lb1[i+1]+lb3[i+1], ld[i]+lf1[i]+lf3[i], ld[i]+lf1[i]+lf2[i]+lf3[i]+lb3[i+1])
      denum =  float_max8(0 , lf2[i]+lb3[i+1], lf1[i]+lf3[i]+lb1[i+1], lf1[i]+lf2[i]+lf3[i]+lb1[i+1]+lb3[i+1], ld[i]+lf3[i]+lb1[i+1]+lb2[i+1], ld[i]+lf2[i]+lf3[i]+lb1[i+1]+lb2[i+1]+lb3[i+1], ld[i]+lf1[i]+lb2[i+1], ld[i]+lf1[i]+lf2[i]+lb2[i+1]+lb3[i+1])
      lc[i]  = num - denum


