""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cimport cython

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


cdef extern from "Sparse_Turbo_lib.h":
  void sparse_turbo_decode(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta)
  void sparse_turbo_decode_MS(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta)
  void sparse_turbo_decode_MS_zero(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta)
  void update_FSM_sparse_eff(long Aq, long N, double thresh, double *pdv, double *pcv, double *out)
  void update_FSM_sparse_eff_test(long Aq, long N, double thresh, double *pdv, double *pcv)

def update_FSM_sparse_eff_func_test(np.ndarray[double, ndim=1, mode="c"] pc not None,
                   np.ndarray[double, ndim=1, mode="c"] pd not None, 
                   np.ndarray[double, ndim=1, mode="c"] res, 
                   Aq, N, theta):
    #cdef double *r
    update_FSM_sparse_eff_test  (Aq, N, theta, <double*> np.PyArray_DATA(pd), <double*>
    np.PyArray_DATA(pc))

def update_FSM_sparse_eff_func(np.ndarray[double, ndim=1, mode="c"] pc not None,
                   np.ndarray[double, ndim=1, mode="c"] pd not None,
                   np.ndarray[double, ndim=1, mode="c"] res not None,
                   Aq, N, theta):
    update_FSM_sparse_eff  (Aq, N, theta, <double*> np.PyArray_DATA(pd), <double*> np.PyArray_DATA(pc),
                 <double*> np.PyArray_DATA(res))

def turbo_decode_func(np.ndarray[double, ndim=1, mode="c"] pc1 not None,
                   np.ndarray[double, ndim=1, mode="c"] pc2 not None,
                   np.ndarray[double, ndim=1, mode="c"] pd not None,
                   np.ndarray[long, ndim=1, mode="c"] res not None,
                   np.ndarray[long, ndim=1, mode="c"] inter not None, 
                   np.ndarray[long, ndim=1, mode="c"] deinter not None, 
                   K, N, Aq, Nc, Nd, Ns, theta):
    sparse_turbo_decode  (<double*> np.PyArray_DATA(pc1), <double*> np.PyArray_DATA(pc2),
                <double*> np.PyArray_DATA(pd), <long*> np.PyArray_DATA(res),
                <long*> np.PyArray_DATA(inter), <long*> np.PyArray_DATA(deinter),
                K, N, Aq, Nd, Nc, Ns, theta)

def turbo_decodeMS_func(np.ndarray[double, ndim=1, mode="c"] pc1 not None,
                   np.ndarray[double, ndim=1, mode="c"] pc2 not None,
                   np.ndarray[double, ndim=1, mode="c"] pd not None,
                   np.ndarray[long, ndim=1, mode="c"] res not None,
                   np.ndarray[long, ndim=1, mode="c"] inter not None, 
                   np.ndarray[long, ndim=1, mode="c"] deinter not None, 
                   K, N, Aq, Nc, Nd, Ns, theta):
    sparse_turbo_decode_MS  (<double*> np.PyArray_DATA(pc1), <double*> np.PyArray_DATA(pc2),
                <double*> np.PyArray_DATA(pd), <long*> np.PyArray_DATA(res),
                <long*> np.PyArray_DATA(inter), <long*> np.PyArray_DATA(deinter),
                K, N, Aq, Nd, Nc, Ns, theta)

def turbo_decodeMS_zero_func(np.ndarray[double, ndim=1, mode="c"] pc1 not None,
                   np.ndarray[double, ndim=1, mode="c"] pc2 not None,
                   np.ndarray[double, ndim=1, mode="c"] pd not None,
                   np.ndarray[long, ndim=1, mode="c"] res not None,
                   np.ndarray[long, ndim=1, mode="c"] inter not None, 
                   np.ndarray[long, ndim=1, mode="c"] deinter not None, 
                   K, N, Aq, Nc, Nd, Ns, theta):
    sparse_turbo_decode_MS_zero  (<double*> np.PyArray_DATA(pc1), <double*> np.PyArray_DATA(pc2),
                <double*> np.PyArray_DATA(pd), <long*> np.PyArray_DATA(res),
                <long*> np.PyArray_DATA(inter), <long*> np.PyArray_DATA(deinter),
                K, N, Aq, Nd, Nc, Ns, theta)

@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_sparse_encode(np.ndarray[long, ndim=1, mode="c"] data not None,
                      np.ndarray[long, ndim=1, mode="c"] out not None, int Aq):
    cdef unsigned int N = <unsigned int> len(data)
    cdef unsigned int i
    cdef unsigned int s0 = 0
    cdef unsigned int s1 = 0
    cdef unsigned int s2 = 0
    cdef unsigned int temp
    for i in range(N):
        out[i] = s0 ^ s2 ^ data[i]
#        temp = s0
#        s0 = s1
#        s1 = temp ^ s2
#        s2 = s2 ^ data[i]
        temp = s1
        s1 = s0 ^ s2
        s0 = s2 ^ data[i]
        s2 = temp
         
                      
@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_sparse_encode_MS(np.ndarray[long, ndim=1, mode="c"] data not None,
                      np.ndarray[long, ndim=1, mode="c"] out not None, int Aq):
    cdef unsigned int N = <unsigned int> len(data)
    cdef unsigned int i
    cdef unsigned int s0 = 0
    cdef unsigned int s1 = 0
    cdef unsigned int s2 = 0
    cdef unsigned int temp
    for i in range(N):
        out[i] = (s0 + s2 + data[i]) % Aq
#        temp = s0
#        s0 = s1
#        s1 = temp ^ s2
#        s2 = s2 ^ data[i]
        temp = s1
        s1 = (s0 + s2) % Aq
        s0 = (s2 + data[i]) % Aq
        s2 = temp
#        print s0,s1,s2, s2*Aq**2+s1*Aq+s0

def FSM_sparse_encode_MS_simple(np.ndarray[long, ndim=1, mode="c"] data not None,
                      np.ndarray[long, ndim=1, mode="c"] out not None, int Aq):
    cdef unsigned int N = <unsigned int> len(data)
    cdef unsigned int i
    cdef unsigned int s = 0
    for i in range(N):
        out[i] = (s + data[i]) % Aq
        s = out[i] 

@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_update_ref_simple_Rxx(np.ndarray[double, ndim=2, mode="c"] pd not None,
                       np.ndarray[double, ndim=2, mode="c"] pc not None,
                       np.ndarray[double, ndim=2, mode="c"] pout not None,
                       np.ndarray[double, ndim=2, mode="c"] Rxx not None):
    cdef unsigned int N = <unsigned int> pd.shape[0]
    cdef unsigned int Aq = <unsigned int> pd.shape[1]
    cdef unsigned int i
    cdef unsigned int di
    cdef unsigned int ci
    cdef unsigned int si
    cdef unsigned int Si
    cdef unsigned int t
    cdef unsigned int t1
    # Forward messages initialization
    sf = np.zeros([N + 1, Aq], float)
    sf[0,0] = 1 # zero init state
    # Backward messages initialization
    sb = np.ones([N + 1, Aq], float)/Aq
    ## Forward 
    for i in range(0,N):
      for si in range(Aq):
        if (sf[i, si] > 1e-6):  
          for di in range(Aq):  
            if (pd[i, di] > 1e-6):  
              ci = (si + di) % Aq
              sf[i+1, ci] += sf[i, si] * pd[i, di] * pc[i, ci]
      # Norming
      sf[i+1,:] = sf[i+1,:]/sf[i+1,:].sum()
      vec = np.asarray([sf[i+1,:]])
      Rxx += np.dot(vec.T,vec)/float(N)

    ## Backward 
    for i in range(N-1,-1,-1):
      for Si in range(Aq):
        if (sb[i + 1, Si] > 1e-6):  
          ci = Si  
          for di in range(Aq):
            if (pd[i, di] > 1e-6):  
              sb[i, (ci - di + Aq)%Aq] += pd[i, di] * pc[i, ci] * sb[i+1, Si]
      # Norming                
      sb[i,:] = sb[i,:]/sb[i,:].sum()
      vec = np.asarray([sb[i+1,:]])
      Rxx += np.dot(vec.T,vec)/float(N)

    ## To data
    for i in range(0,N):
      for si in range(Aq):
        if (sf[i, si] > 1e-6):  
          for di in range(Aq):  
            if (pd[i, di] > 1e-6):  
              ci = (di + si) % Aq   
              Si = ci
              pout[i,di] += sb[i+1,Si]  * pc[i, ci] * sf[i,si]
      # Norming                
      pout[i,:] = pout[i,:]/pout[i,:].sum()



@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_update_eff(np.ndarray[double, ndim=2, mode="c"] pd not None,
                       np.ndarray[double, ndim=2, mode="c"] pc not None,
                       np.ndarray[double, ndim=2, mode="c"] pout not None, thresh):
    '''
    '''
    cdef unsigned int N = <unsigned int> pd.shape[0]
    cdef unsigned int Aq = <unsigned int> pd.shape[1]
    cdef unsigned int i
    cdef unsigned int di
    cdef unsigned int ci
    cdef unsigned int s0i
    cdef unsigned int s1i
    cdef unsigned int s2i
    cdef unsigned int si
    cdef unsigned int S0i
    cdef unsigned int S1i
    cdef unsigned int S2i
    cdef unsigned int Si
    cdef unsigned int t
    cdef unsigned int t1
    # Forward marginalized messages initialization
    s0f = np.zeros([N + 1, Aq], float)
    s1f = np.zeros([N + 1, Aq], float)
    s2f = np.zeros([N + 1, Aq], float)
    s0f[0,0] = 1 # zero init state
    s1f[0,0] = 1 # zero init state
    s2f[0,0] = 1 # zero init state
    # Backward marginalized messages initialization
    s0b = np.ones([N + 1, Aq], float)/Aq
    s1b = np.ones([N + 1, Aq], float)/Aq
    s2b = np.ones([N + 1, Aq], float)/Aq
    # Full message initialization
    sf = np.zeros([N + 1, Aq**3], float)
    sf[0,0] = 1
    sb = np.zeros([N + 1, Aq**3], float)
    sb[N,:] = np.ones(Aq**3, float)*(1./Aq**3)
    ## Forward 
    for i in range(0,N):
      for s2i in range(Aq):
        if s2f[i, s2i] > thresh:
          for s0i in range(Aq):
            if s0f[i, s0i] > thresh:  
              for di in range(Aq):
                ci = (s0i + s2i + di) % Aq
                if ((pd[i, di] > thresh) and (pc[i,ci] > thresh)):
                  for s1i in range(Aq):
                    if s1f[i, s1i] > thresh:  
                      S0i = (di + s2i) % Aq  
                      S1i = (s0i + s2i) % Aq
                      S2i = s1i
                      Si = S2i*Aq**2 + Aq * S1i + S0i
                      si = s2i*Aq**2 + Aq * s1i + s0i
                      val = sf[i, si] * pd[i, di] * pc[i, ci]
                      #val = s2f[i, s2i] * s0f[i, s0i] * s1f[i, s0i] * pd[i, di] * pc[i, ci]
                      sf[i + 1, Si] += val
                      s2f[i + 1, S2i] += val
                      s1f[i + 1, S1i] += val
                      s0f[i + 1, S0i] += val
      # Norming
      s = sf[i+1, :].sum()
      if s > 0:
        sf[i+1, :] = sf[i+1, :] / s
        s0f[i+1,:] = s0f[i+1,:]/s0f[i+1,:].sum()
        s1f[i+1,:] = s1f[i+1,:]/s1f[i+1,:].sum()
        s2f[i+1,:] = s2f[i+1,:]/s2f[i+1,:].sum()
      else:
        print 'lost track - forward'  
        sf[i+1, :] = np.ones(Aq**3)/float(Aq**3)
        s0f[i+1,:] = np.ones(Aq)/float(Aq)
        s1f[i+1,:] = np.ones(Aq)/float(Aq)
        s2f[i+1,:] = np.ones(Aq)/float(Aq)
    

    ## Backward 
    for i in range(N-1,-1,-1):
     # print '--------i:%d-----------------'%i
      for S1i in range(Aq):
        if s1b[i+1, S1i] > thresh:
          for di in range(Aq):
            ci = (S1i + di) % Aq
            if ((pd[i, di] > thresh) and (pc[i,ci] > thresh)):
              for S0i in range(Aq):
                if s0b[i+1, S0i] > thresh:  
                  for S2i in range(Aq):
                    if s2b[i+1, S2i] > thresh:  
                      #s0i = (Aq + S1i - S0i + di) % Aq  
                      #s1i = S2i
                      #s2i = (Aq + S0i - di) % Aq
                      s0i = (Aq + ci - S0i) % Aq  
                      s1i = S2i
                      s2i = (Aq + S1i + S0i - ci) % Aq
                      Si = S2i*Aq**2 + Aq * S1i + S0i
                      si = s2i*Aq**2 + Aq * s1i + s0i
                      val = sb[i+1, Si] * pd[i, di] * pc[i, ci]
                      #print si,Si,s0i,s1i,s2i,S0i,S1i,S2i,di,ci,val
                      sb[i , si] += val
                      s2b[i , s2i] += val
                      s1b[i , s1i] += val
                      s0b[i , s0i] += val
      # Norming
      s = sb[i, :].sum()
      if s > 0:
        sb[i, :] = sb[i, :] / s
        s0b[i,:] = s0b[i,:]/s0b[i,:].sum()
        s1b[i,:] = s1b[i,:]/s1b[i,:].sum()
        s2b[i,:] = s2b[i,:]/s2b[i,:].sum()
      else:
        print 'lost track - backward'  
        sb[i, :] = np.ones(Aq**3)/float(Aq**3)
        s0b[i,:] = np.ones(Aq)/float(Aq)
        s1b[i,:] = np.ones(Aq)/float(Aq)
        s2b[i,:] = np.ones(Aq)/float(Aq)

    ## To data
    for i in range(0,N):
      for s1i in range(Aq):          
        S2i = s1i  
        if ((s1f[i, s1i] > thresh) and (s2b[i+1, S2i] > thresh)):
          for s0i in range(Aq):
            if s0f[i, s0i] > thresh:  
              for s2i in range(Aq):
                S1i = (s0i + s2i) % Aq
                if ((s2f[i, s2i] > thresh) and (s1b[i+1,S1i] > thresh)):
                  for ci in range(Aq):
                    di = (Aq*2 + ci - s0i - s2i) % Aq
                    S0i = (di + s2i) % Aq
                    if ((pc[i, ci] > thresh) and (s0b[i + 1, S0i] > thresh)):
                      Si = S2i*Aq**2 + Aq * S1i + S0i
                      si = s2i*Aq**2 + Aq * s1i + s0i
                      val = pc[i,ci] * sb[i+1,Si] * sf[i, si]                        
                      pout[i, di] += val   
      # Norming
      s = pout[i, :].sum()
      if s > 0:
        pout[i, :] = pout[i, :] / s
      else:
        print 'lost track - data'  
        pout[i,:] = np.ones(Aq)/float(Aq)


@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_update_ref_Rxx(np.ndarray[double, ndim=2, mode="c"] pd not None,
                       np.ndarray[double, ndim=2, mode="c"] pc not None,
                       np.ndarray[double, ndim=2, mode="c"] pout not None,
                       np.ndarray[double, ndim=2, mode="c"] Rxx not None):
    cdef unsigned int N = <unsigned int> pd.shape[0]
    cdef unsigned int Aq = <unsigned int> pd.shape[1]
    cdef unsigned int i
    cdef unsigned int di
    cdef unsigned int ci
    cdef unsigned int s0i
    cdef unsigned int s1i
    cdef unsigned int s2i
    cdef unsigned int S0i
    cdef unsigned int S1i
    cdef unsigned int S2i
    cdef unsigned int t
    cdef unsigned int t1
    # Forward messages initialization
    s0f = np.zeros([N + 1, Aq], float)
    s1f = np.zeros([N + 1, Aq], float)
    s2f = np.zeros([N + 1, Aq], float)
    s0f[0,0] = 1 # zero init state
    s1f[0,0] = 1 # zero init state
    s2f[0,0] = 1 # zero init state
    # Backward messages initialization
    s0b = np.ones([N + 1, Aq], float)/Aq
    s1b = np.ones([N + 1, Aq], float)/Aq
    s2b = np.ones([N + 1, Aq], float)/Aq
    ## Forward 
    for i in range(0,N):
      for s1i in range(Aq):
        s2f[i+1, s1i] = s1f[i, s1i]  
      for s2i in range(Aq):
        for s0i in range(Aq):
          t = (s0i + s2i) % Aq  
          for di in range(Aq):  
            ci = (t + di) % Aq
            s1f[i+1, t] += s0f[i, s0i] * s2f[i, s2i] * pd[i, di] * pc[i, ci]
            t1 = (di + s2i) % Aq
            ci = (t1 + s0i) % Aq
            s0f[i+1, t1] += s2f[i, s2i] * pd[i, di] * s0f[i, s0i] * pc[i, ci]
      # Norming
      s0f[i+1,:] = s0f[i+1,:]/s0f[i+1,:].sum()
      s1f[i+1,:] = s1f[i+1,:]/s1f[i+1,:].sum()
      s2f[i+1,:] = s2f[i+1,:]/s2f[i+1,:].sum()
      sf = np.asarray([s0f[i+1, s0i] * s1f[i+1, s1i] * s2f[i+1, s2i] for s0i in range(Aq) \
                  for s1i in range(Aq) for s2i in range(Aq)])
      print 'sf:', sf/np.sum(sf)

    ## Backward 
    for i in range(N-1,-1,-1):
      for S2i in range(Aq):
        s1b[i, S2i] = s2b[i+1, S2i]  
      for S0i in range(Aq):
        for di in range(Aq):
          s2b[i, (S0i - di + Aq)%Aq] += s0b[i+1, S0i] * pd[i, di]
          for S1i in range(Aq):
            ci = (S1i + di) % Aq  
            val = pd[i, di] * s0b[i+1, S0i] * pc[i, ci] * s1b[i+1, S1i]
            s0b[i, (ci - S0i + Aq)%Aq] += val
      # Norming                
      s0b[i,:] = s0b[i,:]/s0b[i,:].sum()
      s1b[i,:] = s1b[i,:]/s1b[i,:].sum()
      s2b[i,:] = s2b[i,:]/s2b[i,:].sum()

    ## To data
    for i in range(0,N):
      for s0i in range(Aq):
        for s1i in range(Aq):
          for s2i in range(Aq):
            for di in range(Aq):  
              ci = (di + s2i + s0i) % Aq   
              S0i = (s2i + di) % Aq 
              S1i = (s0i + s2i) % Aq
              S2i = s1i
              pout[i,di] += s0b[i+1,S0i]  * s1b[i+1,S1i] * s2b[i+1,S2i] * pc[i, ci] * \
                            s0f[i,s0i]  * s1f[i,s1i] * s2f[i,s2i]
      # Norming                
      pout[i,:] = pout[i,:]/pout[i,:].sum()


@cython.boundscheck(False) # turn of bounds-checking for entire function
def FSM_update_sparse(np.ndarray[long, ndim=2, mode="c"] pd_inds not None,
                      np.ndarray[double, ndim=2, mode="c"] pd_vals not None,
                      np.ndarray[long, ndim=2, mode="c"] pc_inds not None,
                      np.ndarray[double, ndim=2, mode="c"] pc_vals not None,
                      np.ndarray[long, ndim=2, mode="c"] S not None,
                      np.ndarray[long, ndim=2, mode="c"] Q not None,
                      np.ndarray[long, ndim=2, mode="c"] S1 not None,
                      np.ndarray[long, ndim=2, mode="c"] Q1 not None,
                      np.ndarray[long, ndim=2, mode="c"] Qm1 not None, Ndi, Nsi, Nci):
    cdef unsigned int N = <unsigned int> pd_inds.shape[0]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int ci
    cdef unsigned int ii
    cdef unsigned int si
    cdef unsigned int si1
    cdef unsigned int di
    cdef unsigned int Nd = <unsigned int> Ndi
    cdef unsigned int Ns = <unsigned int> Nsi
    cdef unsigned int Nc = <unsigned int> Nci
    # Forward messages initialization
    sf_vals = np.zeros([N+1, Ns], float)
    sf_inds = np.zeros([N+1, Ns], int)
    sf_vals[0,0] = 1 # zero init state
    sf_inds[0,1] = -1 # Only the first value is active 
    # Backward messages initialization
    sb_vals = np.zeros([N+1, Ns], float)
    sb_inds = np.zeros([N+1, Ns], int)
    sb_vals[N,0] = 1 # zero final state
    sb_inds[N,1] = -1 # Only the first value is active 
    # Messages to data initialization
    pd_out_vals = np.zeros([N, Nd], float)
    pd_out_inds = np.zeros([N, Nd], int)

    vals = np.zeros([2, Q.shape[0]*Q.shape[1]], float) # temporary vector (allocation huge memory,  but usage only as needed)

    ## Forward 
    for i in range(0,N):
      sf_inds[i+1, 0] = -1  # No update by default 
      ii = 0
      di = 0          
      while (di < Nd) and (pd_inds[i,di] != -1):
        si = 0
        while (si < Ns) and (sf_inds[i,si] != -1):
          ci = 0  
          while (ci < Nc) and (pc_inds[i,ci] != -1):
            if (Q[pd_inds[i,di], sf_inds[i,si]] == pc_inds[i,ci]): # Find all compatible codewords
              vals[0,ii] = S[pd_inds[i,di], sf_inds[i,si]] # next state 
              vals[1,ii] = pd_vals[i, di]*sf_vals[i, si]*pc_vals[i, ci] # corresponding prob 
              ii += 1
            ci += 1
          si += 1
        di += 1          
      vec_inds = np.unique(vals[0,:ii])
      vec_vals = np.asarray([vals[1,np.nonzero(vals[0,:ii] == j)[0]].sum() for j in vec_inds])
      vec = np.vstack([vec_inds, vec_vals])
      sf1_sorted = vec[:,vec[1,:].argsort()[-1::-1]] # Sorted according to values
#      print vec, '\n', sf1_sorted,'\n\n'
      if Ns <= len(vec_inds):
        sf_vals[i+1,:] = sf1_sorted[1,:Ns]
        sf_inds[i+1,:] = sf1_sorted[0,:Ns]
      else:  
        sf_vals[i+1,:len(vec_inds)] = sf1_sorted[1,:]
        sf_inds[i+1,:len(vec_inds)] = sf1_sorted[0,:]        
        sf_inds[i+1,len(vec_inds)] = -1
      # Norming
      sf_sum = np.sum(sf_vals[i+1,:])
      if sf_sum > 0:
        sf_vals[i+1,:] = sf_vals[i+1,:]/sf_sum
      else:
        sf_inds[i+1,:] = -1 # no update  


    ## Backward

    for i in np.arange(N-1,-1,-1):
      sb_inds[i, 0] = -1  # No update by default 
      ii = 0
      di = 0          
      if (sb_inds[i+1,0] == -1):
        while (di < Nd) and (pd_inds[i,di] != -1):
          ci = 0
          while (ci < Nc) and (pc_inds[i,ci] != -1):
            if (pd_vals[i, di] * pc_vals[i, ci] > 0): # threshold to reduce complexity 
              states = np.nonzero(Q[pd_inds[i, di], :] == pc_inds[i, ci])[0]
              for si in states:
#                print di, si, ci, pd_vals[i, di]*pc_vals[i, ci]  
                vals[0,ii] = si # previous state 
                vals[1,ii] = pd_vals[i, di]*pc_vals[i, ci] # corresponding prob 
                ii += 1
            ci += 1
          di += 1          
#        print vals  
      else:    
        while (di < Nd) and (pd_inds[i,di] != -1):
          si = 0
          while (si < Ns) and (sb_inds[i+1,si] != -1):
            ci = 0  
            while (ci < Nc) and (pc_inds[i,ci] != -1):
              if (Q1[pc_inds[i,ci], sb_inds[i+1,si]] == pd_inds[i,di]): # Find all compatible data
#                print 'ttt', pc_inds[i,ci],   Q[pd_inds[i,di], S1[pc_inds[i,ci], sb_inds[i+1,si]]]
                vals[0,ii] = S1[pc_inds[i,ci], sb_inds[i+1,si]] # previous state 
                vals[1,ii] = pd_vals[i, di]*sb_vals[i+1, si]*pc_vals[i, ci] # corresponding prob 
#                print vals[1,ii], pd_vals[i, di],sb_vals[i+1, si],pc_vals[i, ci]
#                print vals[0,ii], pc_inds[i,ci], sb_inds[i+1,si] # previous state 
                ii += 1
              ci += 1
            si += 1
          di += 1          
#      print 'Vals:', vals

      vec_inds = np.unique(vals[0,:ii])
      vec_vals = np.asarray([vals[1,np.nonzero(vals[0,:ii] == j)[0]].sum() for j in vec_inds])
      vec = np.vstack([vec_inds, vec_vals])
#      print i, vec
      sb1_sorted = vec[:,vec[1,:].argsort()[-1::-1]] # Sorted according to values
      if Ns <= len(vec_inds):
        sb_vals[i,:] = sb1_sorted[1,:Ns]
        sb_inds[i,:] = sb1_sorted[0,:Ns]
      else:  
        sb_vals[i,:len(vec_inds)] = sb1_sorted[1,:]
        sb_inds[i,:len(vec_inds)] = sb1_sorted[0,:]        
        sb_inds[i,len(vec_inds)] = -1
      # Norming
      sb_sum = np.sum(sb_vals[i,:])
      if sb_sum > 0:
        sb_vals[i,:] = sb_vals[i,:]/sb_sum
#      print i, sb_vals[i,:], sb_vals[i+1,:]
#    print sf_inds, sf_vals
#    print sb_inds, sb_vals

    ## To Data
    for i in range(0,N):
      if (sb_inds[i+1, 0] + sf_inds[i, 0] + pc_inds[i, 0] < -1):  # No data no update
        pd_out_inds[i, 0] = -1
        continue # skip the rest of the update
      elif sb_inds[i+1, 0] == -1: # Available forward message and codewords
        ii = 0
        ci = 0          
        while (ci < Nc) and (pc_inds[i,ci] != -1):
          si = 0
          while (si < Ns) and (sf_inds[i,si] != -1):
            vals[0,ii] = Qm1[pc_inds[i,ci], sf_inds[i,si]] # compatible data
            vals[1,ii] = pc_vals[i, ci]*sf_vals[i, si] # corresponding prob 
            ii += 1
            si += 1
          ci += 1          
      elif sf_inds[i, 0] == -1: # Available backward message and codewords
        ii = 0
        ci = 0          
        while (ci < Nc) and (pc_inds[i,ci] != -1):
          si = 0
          while (si < Ns) and (sb_inds[i+1,si] != -1):
            vals[0,ii] = Q1[pc_inds[i,ci], sb_inds[i+1,si]] # compatible data
            vals[1,ii] = pc_vals[i, ci]*sb_vals[i+1, si] # corresponding prob 
            ii += 1
            si += 1
          ci += 1          
      else:  # Available all messages        
        ii = 0
        ci = 0          
        while (ci < Nc) and (pc_inds[i,ci] != -1):
          si = 0
          while (si < Ns) and (sf_inds[i,si] != -1):
            si1 = 0  
            while (si1 < Ns) and (sb_inds[i+1,si1] != -1):
              if (S1[pc_inds[i,ci], sb_inds[i+1,si1]] == sf_inds[i,si]): # Find all compatible forward indices
#                print ii, ci, si1
                vals[0,ii] = Q1[pc_inds[i,ci], sb_inds[i+1,si1]] # corresponding data index 
                vals[1,ii] = pc_vals[i, ci]*sf_vals[i, si]*sb_vals[i+1, si1] # corresponding prob 
                ii += 1
              si1 += 1
            si += 1
          ci += 1          

      vec_inds = np.unique(vals[0,:ii])
      vec_vals = np.asarray([vals[1,np.nonzero(vals[0,:ii] == j)[0]].sum() for j in vec_inds])
      vec = np.vstack([vec_inds, vec_vals])
      pd_sorted = vec[:,vec[1,:].argsort()[-1::-1]] # Sorted according to values
      if Nd <= len(vec_inds):
        pd_out_vals[i,:] = pd_sorted[1,:Nd]
        pd_out_inds[i,:] = pd_sorted[0,:Nd]
      else:  
        pd_out_vals[i,:len(vec_inds)] = pd_sorted[1,:]
        pd_out_inds[i,:len(vec_inds)] = pd_sorted[0,:]        
        pd_out_inds[i,len(vec_inds)] = -1
      # Norming
      pd_sum = np.sum(pd_out_vals[i,:])
      if pd_sum > 0:
        pd_out_vals[i,:] = pd_out_vals[i,:]/pd_sum
      else:
        pd_out_inds[i,:] = -1 # no update  
#    print sf_inds
#    print sf_vals
#    print sb_inds
#    print sb_vals

    arg_vals_maxs = np.argmax(sf_vals,axis=1)
#    print [sf_inds[i,arg_vals_maxs[i]] for i in range(N+1)] -- print forward states
    return pd_out_vals, pd_out_inds      

#    print sf_vals+1
@cython.boundscheck(False) # turn of bounds-checking for entire function



def FSM_update_sparse_test(np.ndarray[long, ndim=2, mode="c"] pd_inds not None,
                      np.ndarray[double, ndim=2, mode="c"] pd_vals not None,
                      np.ndarray[long, ndim=2, mode="c"] pc_inds not None,
                      np.ndarray[double, ndim=2, mode="c"] pc_vals not None,
                      np.ndarray[long, ndim=2, mode="c"] S not None,
                      np.ndarray[long, ndim=2, mode="c"] Q not None,
                      np.ndarray[long, ndim=2, mode="c"] S1 not None,
                      np.ndarray[long, ndim=2, mode="c"] Q1 not None,
                      np.ndarray[long, ndim=2, mode="c"] Qm1 not None, Ndi, Nsi, Nci):
    cdef unsigned int N = <unsigned int> pd_inds.shape[0]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int ci
    cdef unsigned int ii
    cdef unsigned int si
    cdef unsigned int si1
    cdef unsigned int di
    cdef unsigned int Nd = <unsigned int> Ndi
    cdef unsigned int Ns = <unsigned int> Nsi
    cdef unsigned int Nc = <unsigned int> Nci
    # Forward messages initialization
    sf_vals = np.zeros([N+1, Ns], float)
    sf_inds = np.zeros([N+1, Ns], int)
    sf_vals[0,0] = 1 # zero init state
    sf_inds[0,1] = -1 # Only the first value is active 
    # Backward messages initialization
    sb_vals = np.zeros([N+1, Ns], float)
    sb_inds = np.zeros([N+1, Ns], int)
    sb_vals[N,0] = 1 # zero final state
    sb_inds[N,1] = -1 # Only the first value is active 
    # Messages to data initialization
    pd_out_vals = np.zeros([N, Nd], float)
    pd_out_inds = np.zeros([N, Nd], int)

    vals = np.zeros([2, Q.shape[0]*Q.shape[1]], float) # temporary vector (allocation huge memory,  but usage only as needed)
    v1 = np.zeros([ Q.shape[0]*Q.shape[1]], int) # temporary vector (allocation huge memory,  but usage only as needed)
    v2 = np.zeros([ Q.shape[0]*Q.shape[1]], float) # temporary vector (allocation huge memory,  but usage only as needed)
    
    ## Forward 
    for i in range(0,N):
      sf_inds[i+1, 0] = -1  # No update by default 
      ii = 0
      di = 0          
      while (di < Nd) and (pd_inds[i,di] != -1):
        si = 0
        while (si < Ns) and (sf_inds[i,si] != -1):
          ci = 0  
          while (ci < Nc) and (pc_inds[i,ci] != -1):
            if (Q[pd_inds[i,di], sf_inds[i,si]] == pc_inds[i,ci]): # Find all compatible codewords
              v1[ii] = S[pd_inds[i,di], sf_inds[i,si]] # next state 
              v2[v1[ii]] = pd_vals[i, di]*sf_vals[i, si]*pc_vals[i, ci] # corresponding prob 
              ii += 1
            ci += 1
          si += 1
        di += 1          
      vec_inds = np.unique(vals[0,:ii])
      l_inds = <unsigned int> len(vec_inds)
      vec_vals = np.asarray([vals[1,np.nonzero(vals[0,:ii] == j)[0]].sum() for j in vec_inds])
      vec = np.vstack([vec_inds, vec_vals])
      sf1_sorted = vec[:,vec[1,:].argsort()[-1::-1]] # Sorted according to values
##      print vec, '\n', sf1_sorted,'\n\n'
      if Ns <= l_inds:
#        for j in range(Ns):
#          sf_vals[i+1,j] = sf1_sorted[1,j]
#          sf_inds[i+1,j] = sf1_sorted[0,j]        
        sf_vals[i+1,:] = sf1_sorted[1,:Ns]
        sf_inds[i+1,:] = sf1_sorted[0,:Ns]
      else:  
#        for j in range(l_inds):
#          sf_vals[i+1,j] = sf1_sorted[1,j]
#          sf_inds[i+1,j] = sf1_sorted[0,j]        
        sf_vals[i+1,:l_inds] = sf1_sorted[1,:]
        sf_inds[i+1,:l_inds] = sf1_sorted[0,:]        
        sf_inds[i+1,l_inds] = -1
#      # Norming
      sf_sum = np.sum(sf_vals[i+1,:])
      if sf_sum > 0:
        sf_vals[i+1,:] = sf_vals[i+1,:]/sf_sum
      else:
        sf_inds[i+1,:] = -1 # no update  

