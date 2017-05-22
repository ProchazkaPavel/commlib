import matplotlib.pyplot as plt
import numpy as np
import os, sys, inspect
sys.path.insert(0,'../LDPC_lib')
sys.path.insert(1,'../EXIT_chats')
import _LDPC_dec_lib
import _GFq_LDPC_lib
import LDPC_lib
import GFq_LDPC_lib
from EXIT_lib import evaluate_extrinsic_info_NI, generate_samples, plot_EXIT, plot_dec
from rew import *
#from numba import jit
from joblib import Parallel, delayed
import _Turbo_lib
import time
from Turbo_lib import FSM, Turbo_Coder
import _Sparse_Turbo_lib 

def VN2_update(Ixi, Iyi, Vxi, Vyi):
    '''
      Variable node update for sparse vectors. See the sparse vectors conventions

      Parameters:
      ----------
      Ixi : numpy array of ints
         List of indices corresponding to the first VN
      Iyi : numpy array of ints
         List of indices corresponding to the second VN
      Vxi : numpy array of floats
         List of values related to indecis to the first VN
      Vyi : numpy array of floats
         List of values related to indecis to the second VN
    '''
    if Ixi[0] == -1:
        return Iyi, Vyi
    if Iyi[0] == -1:
        return Ixi, Vxi
    X = np.vstack([Ixi,Vxi])
    Y = np.vstack([Iyi,Vyi])
    Xs = X[:,X[0,:].argsort()]
    Ys = Y[:,Y[0,:].argsort()]
    Ix = Xs[0,:]
    Iy = Ys[0,:]
    Vx = Xs[1,:]
    Vy = Ys[1,:]
    Iz = np.zeros(len(Ix), int)
    Vz = np.zeros(len(Ix), float)
    iX = 0
    iY = 0
    cnt = 0
    while (iX < len(Ix)) and (iY < len(Iy)):
        if Ix[iX] > Iy[iY]:
            iY += 1
        elif Iy[iY] > Ix[iX]:
            iX += 1
        elif Iy[iY] == Ix[iX]:    
            Iz[cnt] = Ix[iX]
            Vz[cnt] = Vx[iX] * Vy[iY]
            iY += 1
            iX += 1
            cnt += 1
    if cnt < len(Ix) - 1:
        Iz[cnt] = -1
    return Iz, Vz 

def VN3_update(Ixi, Iyi,Izi, Vxi, Vyi, Vzi):
    '''
      Update of sparse vectors for 3 variable nodes. It is also possible to concatenate
      VN2_update(VN2_update(x,y), z), but usage of this is more efficient.
    '''
    if Izi[0] == -1:
        return VN2_update(Ixi, Iyi, Vxi, Vyi)
    if Iyi[0] == -1:
        return VN2_update(Ixi, Izi, Vxi, Vzi)
    if Ixi[0] == -1:
        return VN2_update(Izi, Iyi, Vzi, Vyi)
        
    X = np.vstack([Ixi,Vxi])
    Y = np.vstack([Iyi,Vyi])
    Z = np.vstack([Izi,Vzi])
    Xs = X[:,X[0,:].argsort()]
    Ys = Y[:,Y[0,:].argsort()]
    Zs = Z[:,Z[0,:].argsort()]
    Ix = Xs[0,:]
    Iy = Ys[0,:]
    Iz = Zs[0,:]    
    Vx = Xs[1,:]    
    Vy = Ys[1,:]
    Vz = Zs[1,:]   
    Iu = np.zeros(len(Ix), int)
    Vu = np.zeros(len(Ix), float)
    iX = 0
    iY = 0
    iZ = 0
    cnt = 0
    while (iX < len(Ix)) and (iY < len(Iy) and (iZ < len(Iz))):
        x = Ix[iX]
        y = Iy[iY]
        z = Iz[iZ]
        if (x > y) and (y > z):
            iZ += 1
        if (x > y) and (y == z):            
            iZ += 1
            iY += 1
        if (x > y) and (y < z):            
            iY += 1

        if (x == y) and (y > z):
            iZ += 1
        if (x == y) and (y == z):  
            Iu[cnt] = Ix[iX]
            Vu[cnt] = Vx[iX] * Vy[iY] * Vz[iZ]
            cnt += 1
            iY += 1
            iX += 1            
            iZ += 1        
        
        if (x == y) and (y < z):            
            iY += 1            
            iX += 1 
            
        if (x < y) and (y > z):
            if x < z:
                iX += 1
            elif x == z:
                iZ += 1
                iX += 1
            elif x > z:    
                iZ += 1
        if (x < y) and (y == z):            
            iX += 1            
        if (x < y) and (y < z):            
            iX += 1          
    
    
    if cnt < len(Ix) - 1:
        Iu[cnt] = -1
    return Iu, Vu 
   

class advanced_FSM(FSM):
  '''
  Adding some advanced features on standard FSM functionalities
  '''
  def __init__(self, S = [], Q = [], S1 = [], Q1 = [], Qm1 = []):
    FSM.__init__(self, S, Q)
    self.K = int(np.round(np.log(self.Ms)/np.log(self.Md))) # Constraint length
    self.Mc = np.max(self.Q) + 1 # Cardinality of codewords
    self.S1 = S1
    self.Q1 = Q1
    self.Qm1 = Qm1

  def term_finder(self, act_level, s):
    '''
      Function to find the termination data sequence such that the final state is the zero one
    '''
    if act_level == self.K:
      if s == 0:
        return np.array([], int)
      else:
        return None 
    else:    
      for d in range(self.Md):  
        res = self.term_finder(act_level + 1, self.S[d,s])
        if res is not None:
          return np.hstack([int(d), res])
      return None           

   
  def encode_zero_term(self, data, s0 = 0, t = True): # data 
    '''
    Encode the data such that the zero terminating state is achieved by adding a proper
    non-informational data sequence.
    '''
    s_act = s0 # initial state
    out = np.zeros(len(data)+self.K, int)
    for i in range(0, len(data)):
      d_act = data[i]
      out[i] = self.Q[d_act, s_act]
      s_act = self.S[d_act, s_act] 
#      print s_act
    d_add = self.term_finder(0, s_act)  # additional non-informative data securing the zero final state
    for i in range(len(data), len(data)+self.K):
      d_act = d_add[i-len(data)]
      out[i] = self.Q[d_act, s_act]
      s_act = self.S[d_act, s_act]    
 #     print s_act
 #   print data, d_add  
 #   print out
    return out, d_add
  


  def update_lim(self, pd_inds, pd_vals, pc_inds, pc_vals, Nd, Ns,Nc):
    '''
      Update FSM with given maximal length of pdf vector:

      The pdfs are supposed to be sparse vectors. The FG-SPA runs upon these sparse vectors keeping 
      them sparse. It is designed to efficiently provide decoding for higher order decoders, where
      the length of pdf vectors is very large in the exact evaluation, especially for number of
      states.

      Three stages are assumed:

      Forward:
      --------
                        _____
                       |     |
                       | sn  |
                       |_____|   
                          |
                          | | ps
                          | v
            ____   pd   __|__   pc   ____
           |    |  ->  |     |  <-  |    |
           | dn |------|  F  |------| cn |
           |____|      |_____|      |____|
                          |
                          | | ps1
                          | v
                        __|__  
                       |     |
                       | sn1 |
                       |_____|   
    
      ps[0] refers to the zero initial state (single value)

    Backward:
    ---------
      ps is to be updated. ps1[-1] is initialized over all possible values(!) - can be very complex.
      Improving ideas - take only possible values for ps[-1] or terminate in a know state

    To data:
    --------
    '''
    Q = self.Q
    S = self.S
    Q1 = self.Q1
    S1 = self.S1
    Qm1 = self.Qm1
    N = len(pd_inds)
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
    print sf_inds, sf_vals
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

def pmf2df(pmf, max_vals):
  x,y = pmf.shape
  if y <= max_vals:
    inds = np.array([np.arange(y)]).repeat(x,axis=0)
    vals = pmf
  else:
    temp = np.argsort(pmf, axis = 1)[:,-1::-1]        
    vals = np.vstack([pmf[i,temp[i,:max_vals]] for i in range(x)])        
    inds = temp[:,:max_vals]
      
  return inds, vals

class Sparse_Turbo_Coder(Turbo_Coder):
  def __init__(self, k, rate=1./3, inter=[], deinter=[], K1 = 3, Aq = 2, track = 2, K = 5):
    Turbo_Coder.__init__(self, k, rate, inter, deinter, K1, Aq, 0, track, K)
    self.S = np.zeros([Aq,Aq**3],int)
    self.Q = np.zeros([Aq,Aq**3],int)
    self.S1 = np.zeros([Aq,Aq**3],int)
    self.Q1 = np.zeros([Aq,Aq**3],int)
    self.Qm1 = np.zeros([Aq,Aq**3],int)
    _Turbo_lib.create_matrix_SQ_XOR_all(self.S,self.Q,self.S1,self.Q1,self.Qm1)
    self.F = advanced_FSM(self.S,self.Q,self.S1,self.Q1, self.Qm1)
    self.n = int(round(k / rate)) 
    self.k = k
    self.p = self.n - self.k

  def encode(self, b):
    b_int = b[self.inter]
    k = len(b)
    p = self.n - k
    c1 = np.zeros(k, int)
    _Sparse_Turbo_lib.FSM_sparse_encode(b, c1, self.Aq)
#    print b, c1
    c2 = np.zeros(k, int)
    _Sparse_Turbo_lib.FSM_sparse_encode(b_int, c2, self.Aq)
    cd = np.vstack([c1[self.map_inds_c1[:(p/2)]], c2[self.map_inds_c2]]).flatten(1)
    if (p % 2 == 1): # different length of c1 and c2
      cd = np.hstack([cd, c1[-1]])
    return np.hstack([b, cd]) # systematic mapping 

  def encode_zero(self, b):
    b_int = b[self.inter]
    k = len(b)
    p = self.n - k
    c1_full, b_add1 = self.F.encode_zero_term(b)
    c2_full, b_add2 = self.F.encode_zero_term(b_int)
    c1 = c1_full[:(-self.F.K)]
    c2 = c2_full[:(-self.F.K)]
    c1_add = c1_full[(-self.F.K):]
    c2_add = c2_full[(-self.F.K):]    
#    print c1, c2, self.map_inds_c2
#    t2 = c2[self.map_inds_c2]
#    print t2, self.map_inds_c1[:(p/2)]
#    t1 = c1[self.map_inds_c1[:(p/2)]]
#    print t1
    cd = np.vstack([c1[self.map_inds_c1[:(p/2)]], c2[self.map_inds_c2]]).flatten(1)
    if (p % 2 == 1): # different length of c1 and c2
      cd = np.hstack([cd, c1[-1]])
    return np.hstack([b, b_add1, b_add2, cd, c1_add, c2_add]) # systematic mapping 

  def decode(self, mu_i, max_vals = 40, max_valsS = 100):
    '''
      stream decode - nonoptimized old fashioned - soft out\
      mu_i .... input metric
      K ... number of iterations in decoder
      max_vals ... sparse vector data and codeword represented by max_vals most probable pmf values and corresponding  indecis
      max_valsS ... the same for modulator state
    '''
    k = self.k
    K = self.K
    p = self.p
    Aq = self.Aq
    inter = self.inter
    deinter = self.deinter
    mbi = mu_i[:k,:]
    mci = mu_i[(k+self.F.K*2):(-2*self.F.K),:]
    b1term_i, b1term_v= pmf2df(mu_i[k:(k+self.F.K), :], max_vals)
    b2term_i, b2term_v= pmf2df(mu_i[(k+self.F.K):(k+2*self.F.K), :], max_vals)
    c1term_i, c1term_v= pmf2df(mu_i[(-2*self.F.K):(-self.F.K), :], max_vals)
    c2term_i, c2term_v = pmf2df(mu_i[(-self.F.K):, :], max_vals)
    map_inds_c1 = self.map_inds_c1
    map_inds_c2 = self.map_inds_c2
    mci1t = np.ones([k, Aq], float)/Aq;  
    mci1t[map_inds_c1, :] = mci[::2]
    mci2t = np.ones([k, Aq], float)/Aq;
    mci2t[map_inds_c2, :] = mci[1::2]

    b_inds, b_vals = pmf2df(mbi, max_vals)
    c1_inds, c1_vals = pmf2df(mci1t, max_vals)
    c2_inds, c2_vals = pmf2df(mci2t, max_vals)
    Nd = np.min([max_vals, Aq])
    Nc = np.min([max_vals, Aq])

    c1_inds = np.vstack([c1_inds, c1term_i]) 
    c1_vals = np.vstack([c1_vals, c1term_v])
    c2_inds = np.vstack([c2_inds, c2term_i])
    c2_vals = np.vstack([c2_vals, c2term_v])

    in1_inds = b_inds
    in1_vals = b_vals
    for i in range(0, K):
      in1_inds = np.vstack([in1_inds, b1term_i])
      in1_vals = np.vstack([in1_vals, b1term_v])
      out1_vals, out1_inds = self.F.update_lim(in1_inds, in1_vals, c1_inds, c1_vals, Nd, max_valsS, Nc)
      temp = np.vstack([VN2_update(out1_inds[i,:], b_inds[i,:], out1_vals[i,:], b_vals[i,:]) for i in range(k)])
      in2_inds = temp[::2]
      in2_vals = temp[1::2]
      in2_inds = np.vstack([in2_inds[inter], b2term_i])
      in2_vals = np.vstack([in2_vals[inter], b2term_v])

      out2_vals, out2_inds = self.F.update_lim(in2_inds, in2_vals, c2_inds, c2_vals, Nd, max_valsS, Nc)
      in1_inds = out2_inds[:k][deinter]
      in1_vals = out2_vals[:k][deinter]
      temp = np.vstack([VN2_update(in1_inds[i,:], b_inds[i,:], in1_vals[i,:], b_vals[i,:]) for i in range(k)])
      in1_inds = temp[::2]
      in1_vals = temp[1::2]
    
    temp = np.vstack([VN3_update(in1_inds[i,:], b_inds[i,:], out1_inds[i,:], \
                                 in1_vals[i,:], b_vals[i,:], out1_vals[i,:] ) for i in range(k)])
    return temp[::2], temp[1::2] 
 
  def decide(self, inds, vals):
    return [inds[i,np.argmax(vals[i,:])] for i in range(len(inds))] 
    
class Sparse_Turbo_Coder_MS(Turbo_Coder):
  def __init__(self, k, K = 5, rate=1./3, Aq = 2, inter=[], deinter=[]):
    Turbo_Coder.__init__(self, k, rate, inter, deinter, 3, Aq, 0, 0, K)
    self.S = np.zeros([Aq,Aq**3],int)
    self.Q = np.zeros([Aq,Aq**3],int)
    _Turbo_lib.create_matrix_SQ_MS_mod(self.S,self.Q)
    self.F = advanced_FSM(self.S,self.Q)
    self.n = int(round(k / rate)) 
    self.k = k
    self.p = self.n - self.k

  def encode(self, b):
    b_int = b[self.inter]
    k = len(b)
    p = self.n - k
    c1 = np.zeros(k, int)
    _Sparse_Turbo_lib.FSM_sparse_encode_MS(b, c1, self.Aq)
#    print b, c1
    c2 = np.zeros(k, int)
    _Sparse_Turbo_lib.FSM_sparse_encode_MS(b_int, c2, self.Aq)
    cd = np.vstack([c1[self.map_inds_c1[:(p/2)]], c2[self.map_inds_c2]]).flatten(1)
    if (p % 2 == 1): # different length of c1 and c2
      cd = np.hstack([cd, c1[-1]])
    return np.hstack([b, cd]) # systematic mapping 

  def decode(self, mu_i, max_vals = 40, max_valsS = 100):
    k = self.k
    K = self.K
    Aq = self.Aq
    inter = self.inter
    deinter = self.deinter
    mbi = mu_i[:k,:]
    mci = mu_i[k:, :]
    map_inds_c1 = self.map_inds_c1
    map_inds_c2 = self.map_inds_c2
    mci1t = -np.ones([k, Aq], float)/Aq;  
    mci1t[map_inds_c1, :] = mci[::2]
    mci2t = -np.ones([k, Aq], float)/Aq;
    mci2t[map_inds_c2, :] = mci[1::2]

    pc1 = mci1t.flatten()
    pc2 = mci2t.flatten()
    pd = mbi.flatten()
    res = np.zeros(k, int)

    Nc = Aq
    Nd = Aq
    Ns = Aq**3
    theta = 1e-8
    
    _Sparse_Turbo_lib.turbo_decode_func_MS(pc1, pc2, pd, res, inter, deinter, S, k, Aq, Nc, Nd, Ns, theta)
    return res
  
  def encode_zero(self, b):
    b_int = b[self.inter]
    k = len(b)
    p = self.n - k
    c1_full, b_add1 = self.F.encode_zero_term(b)
    c2_full, b_add2 = self.F.encode_zero_term(b_int)
    c1 = c1_full[:(-self.F.K)]
    c2 = c2_full[:(-self.F.K)]
    c1_add = c1_full[(-self.F.K):]
    c2_add = c2_full[(-self.F.K):]    
#    print c1, c2, self.map_inds_c2
#    t2 = c2[self.map_inds_c2]
#    print t2, self.map_inds_c1[:(p/2)]
#    t1 = c1[self.map_inds_c1[:(p/2)]]
#    print t1
    cd = np.vstack([c1[self.map_inds_c1[:(p/2)]], c2[self.map_inds_c2]]).flatten(1)
    if (p % 2 == 1): # different length of c1 and c2
      cd = np.hstack([cd, c1[-1]])
    return np.hstack([b, b_add1, b_add2, cd, c1_add, c2_add]) # systematic mapping 
