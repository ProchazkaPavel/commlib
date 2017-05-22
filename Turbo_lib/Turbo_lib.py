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



class FSM:
  def __init__(self, S = [], Q = []):
    if S == []: # Simple convolutional code by default
      self.S = np.array([[0,1],[1,0]], int)
    else:
      self.S = S # Md times Ms
    if Q == []:
      self.Q = np.array([[0,1],[1,0]], int)
    else:
      self.Q = Q
    (self.Md, self.Ms) = np.shape(self.S) # Cardinality of data and states
    self.K = int(np.log2(self.Ms)) # Constraint length
    self.Mc = np.max(self.Q) + 1 # Cardinality of codewords
  
#  @jit
  def encode(self, data, s0 = 0, t = True): # data 
    s_act = s0 # initial state
    out = np.zeros(data.shape,int)
    for i in range(0, len(data)):
      d_act = data[i]
      out[i] = self.Q[d_act, s_act]
      s_act = self.S[d_act, s_act]
    
    return out

  def encode_eff(self, data, s0 = 0): # Cython function
    out = numpy.zeros(data.shape,int)
    _Turbo_lib.FSM_encode(data, out, self.S, self.Q)
    return out


  def update(self, pc, pd): # provide BJRC algorithm on FSM (pc and pd are flatten soft inputs
    out = np.zeros(pd.shape)
    _Turbo_lib.update_general_FSM_func(pc, pd, out,self.S, self.Q)
    return out

  def update_C(self, pc, pd): # provide BJRC algorithm on FSM (pc and pd are flatten soft inputs
    '''
      Same as in update, but written in Cython instead C (10 times slower)
    '''
    out = np.zeros(pd.shape, float)
    N, Md = np.uint32(pd.shape)
    N, Mc = np.uint32(pc.shape)
    Md, Ms = np.uint32(self.Q.shape)
    Q = np.uint32(self.Q)
    S = np.uint32(self.S)
    _Turbo_lib.update_general_FSM_C(pc, pd, out, S, Q, Md, Ms, Mc, N)
    return out

  def update_both(self, pc, pd): # provide BJRC algorithm on FSM (pc and pd are flatten soft inputs)
    outc = np.zeros(pc.shape)
    outd = np.zeros(pd.shape)
    _Turbo_lib.update_general_FSM_both_func(pc, pd, outc, outd, self.S, self.Q)
    return outc, outd
  
  def update_LLR(self, lc, ld): # provide BJRC algorithm on FSM using LLRv, binary data and code assumed
    '''
      Pure Python:
      S=Q=[[0,1],[1,0]] is supported so far
    '''
    N = len(lc)
    M = np.log2(self.Ms)
    lf = np.zeros(N + 1, float) # LLR forward
    lb = np.zeros(N + 1, float) # LLR backward
    lf[0] = np.ones(M)*(-np.inf) # zero initial state
    lb[-1] = np.zeros(M) # unknown final state
    for i in range(N):
      num = np.array([lf[i]+lc[i], ld[i]+lc[i]])
      denum = np.array([0, ld[i]+lf[i]])
      lf[i+1] = np.max(num)-np.max(denum)

#    for ind in np.nditer(np.arange(len(lc)), order='F', flags=['external_loop']):
    for ind in range(N):
      i = N - 1 - ind
      num = np.array([lb[i+1]+lc[i], ld[i]])
      denum = np.array([0, ld[i]+lb[i+1]+lc[i]])
#      print i, num, denum
      lb[i] = np.max(num)-np.max(denum)    

    num = np.array([lb[1:]+lc, lf[:-1]])
    denum = np.array([np.zeros(N,float),  lf[:-1] +lc + lb[1:]])
    upd = np.max(num,axis=0) - np.max(denum, axis=0) 
#    print 'ld:\t',ld,'\nlc:\t',lc,'\nlb:\t', lb,'\nlf:\t',lf
    return upd

  def decode_hard_decision(self, c, epsilon=1e-5): # provide decoding with hard decision (binary fixed)
    p = np.vstack([1-np.abs(c - epsilon), np.abs(c - epsilon)]) # softing the message
    return (self.update(p.flatten(1), np.ones(len(c)*2)*0.5)[::2] < 0.5) + 1 - 1

  def test(self, SNR = 10, l = 10000, test_LLR = 1):
    '''
    Testing the FSM update (conv code in AWGN channel)
    S = Q = [[0,1],[1,0]] (K=1)
    '''
    d = np.random.randint(2, size=l)
    c = self.encode(d)
    s = 2*c - 1 # BPSK
    (x, sigma2w) = AWGN(s, SNR)
    mu = np.zeros([l, 2])
    mu[:,0] = np.exp(-np.abs(x + 1)**2/sigma2w)
    mu[:,1] = np.exp(-np.abs(x - 1)**2/sigma2w)
    
    pd = np.ones(l*2, float)/2  # no a priory knowledge
    pc = mu.flatten() # observation
    s_c = time.time()
    res_conv = self.update(pc, pd).reshape(l, 2) # conventional prob based solution
    e_c = time.time()
    s_c2 = time.time()
    res_conv2 = self.update_C(mu, np.ones([l,2],float)/2) # conventional prob based solution
    e_c2 = time.time()
#    print res_conv[:10], np.argmax(res_conv, axis=1)[:10], d[:10]
    nerr_c2 = np.sum(np.argmax(res_conv2, axis=1) != d)
    nerr_c = np.sum(np.argmax(res_conv, axis=1) != d)
    nerr_u = np.sum(np.argmax(mu, axis=1) != c)
    
    print 'Nerr uncoded:%d, \t Nerr coded:%d, \t Time ref:%.6f'%(nerr_u, nerr_c, e_c-s_c)
    print 'Nerr coded2:%d, \t Time coded2:%.6f'%(nerr_c2, e_c2-s_c2)
    if test_LLR:
      lc = np.log(mu[:,1]/mu[:,0]) # LLR from channel
      ld = np.zeros(l, float) # no a priory info
      s_L = time.time()
      res_LLR = self.update_LLR(lc, ld)
      e_L = time.time()
      nerr_L = np.sum((np.sign(res_LLR)+1)/2 != d)
       
      res_LLR2 = np.zeros(l, float)
      s_L2 = time.time()
      _Turbo_lib.update_FSM_LLR1(lc, ld, res_LLR2)
      e_L2 = time.time()
      nerr_L2 = np.sum((np.sign(res_LLR2)+1)/2 != d)
      
      z = 2**30
      lc_int = np.uint32(lc * 2**16 + z) # 2**31 -> 0
      ld_int = np.ones(l, np.uint32)*z # no a priory info
      res_LLRint = np.zeros(l, np.uint32)
      s_Lint = time.time()
      _Turbo_lib.update_FSM_LLR1_int(lc_int, ld_int, res_LLRint)
      e_Lint = time.time()
      nerr_Lint = np.sum((np.sign(np.int64(res_LLRint)-z)+1)/2 != d)
      print 'Nerr LLR:%d, \t Time LLR:%.6f, \t Nerr LLR2:%d, \t Time LLR2:%.6f, \t'%(nerr_L, e_L-s_L, nerr_L2, e_L2-s_L2)
      print 'Nerr LLR_uint:%d, \t Time LLR_uint:%.6f'%(nerr_Lint, e_Lint-s_Lint)

      return nerr_u, nerr_c, nerr_L, nerr_Lint
    else:
      return nerr_u, nerr_c   
  
  def test2(self, SNR = 10, l = 10000, test_LLR = 1):
    '''
    Testing the FSM update (conv code in AWGN channel)
    S = [[0, 2, 1, 3],[1, 3, 0, 2]], Q=[[0, 0, 1, 1],[1, 1, 0, 0]] (K=2)
    '''
    d = np.random.randint(2, size=l)
    c = self.encode(d)
    s = 2*c - 1 # BPSK
    (x, sigma2w) = AWGN(s, SNR)
    mu = np.zeros([l, 2])
    mu[:,0] = np.exp(-np.abs(x + 1)**2/sigma2w)
    mu[:,1] = np.exp(-np.abs(x - 1)**2/sigma2w)
    
    pd = np.ones(l*2, float)/2  # no a priory knowledge
    pc = mu.flatten() # observation
    s_c = time.time()
    res_conv = self.update(pc, pd).reshape(l, 2) # conventional prob based solution
    e_c = time.time()
    s_c2 = time.time()
    res_conv2 = self.update_C(mu, np.ones([l,2],float)/2) # conventional prob based solution
    e_c2 = time.time()
#    print res_conv[:10], np.argmax(res_conv, axis=1)[:10], d[:10]
    nerr_c2 = np.sum(np.argmax(res_conv2, axis=1) != d)
    nerr_c = np.sum(np.argmax(res_conv, axis=1) != d)
    nerr_u = np.sum(np.argmax(mu, axis=1) != c)
    
    print 'Nerr uncoded:%d, \t Nerr coded:%d, \t Time ref:%.6f'%(nerr_u, nerr_c, e_c-s_c)
    print 'Nerr coded2:%d, \t Time coded2:%.6f'%(nerr_c2, e_c2-s_c2)
    if test_LLR:
      lc = np.log(mu[:,1]/mu[:,0]) # LLR from channel
      ld = np.zeros(l, float) # no a priory info

      res_LLR = np.zeros(l, float)
      s_L = time.time()
      _Turbo_lib.update_FSM_LLR2(lc, ld, res_LLR)
      e_L = time.time()
      nerr_L = np.sum((np.sign(res_LLR)+1)/2 != d)

      print 'Nerr LLR:%d, \t Time LLR:%.6f\t'%(nerr_L, e_L-s_L)

    return nerr_u, nerr_c, nerr_L
 
  def test3(self, SNR = 10, l = 10000, test_LLR = 1):
    '''
    Testing the FSM update (conv code in AWGN channel)
    S = [[0, 2, 1, 3, 6, 4, 7, 5], [4, 6, 5, 7, 2, 0, 3, 1]]
    Q = [[0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1]]
    K = 3
    '''
    d = np.random.randint(2, size=l)
    c = self.encode(d)
    s = 2*c - 1 # BPSK
    (x, sigma2w) = AWGN(s, SNR)
    mu = np.zeros([l, 2])
    mu[:,0] = np.exp(-np.abs(x + 1)**2/sigma2w)
    mu[:,1] = np.exp(-np.abs(x - 1)**2/sigma2w)
    
    pd = np.ones(l*2, float)/2  # no a priory knowledge
    pc = mu.flatten() # observation
    s_c = time.time()
    res_conv = self.update(pc, pd).reshape(l, 2) # conventional prob based solution
    e_c = time.time()
    s_c2 = time.time()
    res_conv2 = self.update_C(mu, np.ones([l,2],float)/2) # conventional prob based solution
    e_c2 = time.time()
#    print res_conv[:10], np.argmax(res_conv, axis=1)[:10], d[:10]
    nerr_c2 = np.sum(np.argmax(res_conv2, axis=1) != d)
    nerr_c = np.sum(np.argmax(res_conv, axis=1) != d)
    nerr_u = np.sum(np.argmax(mu, axis=1) != c)
    
    print 'Nerr uncoded:%d, \t Nerr coded:%d, \t Time ref:%.6f'%(nerr_u, nerr_c, e_c-s_c)
    print 'Nerr coded2:%d, \t Time coded2:%.6f'%(nerr_c2, e_c2-s_c2)
    if test_LLR:
      lc = np.log(mu[:,1]/mu[:,0]) # LLR from channel
      ld = np.zeros(l, float) # no a priory info

      res_LLR = np.zeros(l, float)
      s_L = time.time()
      _Turbo_lib.update_FSM_LLR3(lc, ld, res_LLR)
      e_L = time.time()
      nerr_L = np.sum((np.sign(res_LLR)+1)/2 != d)

      print 'Nerr LLR:%d, \t Time LLR:%.6f\t'%(nerr_L, e_L-s_L)

    return nerr_u, nerr_c, nerr_L

 
class Turbo_Coder:
  '''
  k ... length of data stream
  rate ... rate = n/k -> rate of the code
  K1 ... Constraint length of the first FSM (so far both FSMs are equal)
  Aq ... Cardinality of data message
  eff ... efficience flag

  Examples: Creation of simple decoder, testing in AWNG and drawing
  C = Turbo_Coder(6, rate=1./3, inter=np.array([1, 4, 2, 0, 5, 3]), deinter=[3, 0, 2, 5, 1, 4], K1 = 1, Aq = 2, eff = 0, track = 2, K = 5); 
  C.test(-3);
  C.draw()
  '''
  def __init__(self, k, rate=1./3, inter=[], deinter=[], K1 = 3, Aq = 2, eff = 1, track = 2, K = 5):
    self.code = self.create_code(Aq, K1) 
    if eff == 1: ## prepare 32-bit vectors for efficient implementation
      self.inter = np.zeros(k, dtype=np.uint32)
      self.deinter = np.zeros(k, dtype=np.uint32)
      self.S = np.zeros(self.code.S.flatten().shape, dtype=np.uint32)
      self.Q = np.zeros(self.code.Q.flatten().shape, dtype=np.uint32)
      self.S[:] = self.code.S.flatten(1)
      self.Q[:] = self.code.Q.flatten(1)
      self.eff = 32 # 32-bit arithmetic for decoding
    else:
      self.inter = np.zeros(k, int)
      self.deinter = np.zeros(k, int)
      self.eff = 64 # 64-bit arithmetic for decoding

    if inter == []: # create random interleaver
      self.inter[:] = np.random.permutation(k)
      for i in range(k):
        self.deinter[self.inter[i]] = i
    else: # load the interleaver
      self.inter[:] = inter
      self.deinter[:] = deinter

    self.rate = rate
    self.n = int(round(k / rate))
    self.k = k
    self.p = self.n - self.k
    self.Aq = Aq
    self.K = K
    self.map_inds2 = np.asarray([2 * i * k / self.p for i in range(self.p/2)])
    self.map_inds_c1 = np.asarray([2 * i * k / self.p for i in range(int(round(self.p/2.)))]) 
    self.map_inds_c2 = np.asarray([2 * i * k / self.p for i in range(self.p/2)])
    
    if track == 1: # flag determining tracking of messages needed for EXIT analysis
      self.track_messages = 1 
      self.i1o2 = np.zeros([k, Aq, K], float) # input to dec 1 (out dec 2)
      self.i2o1 = np.zeros([k, Aq, K], float) # in dec 2, out dec 1
    elif track == 2:  # flag determining tracking of all messages (debug)
      self.track_messages = 2
      self.i1o2 = np.zeros([k, Aq, K], float) # input to dec 1 (out dec 2)
      self.i2o1 = np.zeros([k, Aq, K], float) # in dec 2, out dec 1
      self.s1_f = np.zeros([k + 1, K1**Aq, K], float) # C1 states forward
      self.s1_b = np.zeros([k + 1, K1**Aq, K], float) # C1 states backward
      self.c1i = np.zeros([k, Aq], float) # C1 input !!! assumes c cardinality Aq
      self.c2i = np.zeros([k, Aq], float) # C2 input
      self.c1o = np.zeros([k, Aq], float) # C1 out
      self.c2o = np.zeros([k, Aq], float) # C2 out
      self.bi = np.zeros([k, Aq], float) # B in
      self.bo = np.zeros([k, Aq], float) # B out
    else:  # No message tracking
      self.track_messages = 0

    

 
  def encode(self, b):
    '''
    Turbo encode a particular data stream b. The resulting stream is given by
    c = [b_0, ..., b_k, c1_0, c2_0, c1_i, c2_i, ...], where i depends on particular rate
    '''
    b_int = b[self.inter]
    k = len(b)
    p = self.n - k
    c1 = self.code.encode(b, 0) 
    c2 = self.code.encode(b_int, 0)
#    print c1, c2, self.map_inds_c2
#    t2 = c2[self.map_inds_c2]
#    print t2, self.map_inds_c1[:(p/2)]
#    t1 = c1[self.map_inds_c1[:(p/2)]]
#    print t1
    cd = np.vstack([c1[self.map_inds_c1[:(p/2)]], c2[self.map_inds_c2]]).flatten(1)
    if (p % 2 == 1): # different length of c1 and c2
      cd = np.hstack([cd, c1[-1]])
    return np.hstack([b, cd]) # systematic mapping

  def decode(self, mu_i, thr = 1e-3):
    '''
      stream decode - nonoptimized old fashioned - soft out\
      mu_i .... input metric
      K ... number of iterations in decoder
    '''
    k = self.k
    K = self.K
    p = self.p
    Aq = self.Aq
    inter = self.inter
    deinter = self.deinter
    mbi = mu_i[:k,:]
    mci = mu_i[k:,:]
    map_inds_c1 = self.map_inds_c1
    map_inds_c2 = self.map_inds_c1

    mci1t = np.ones([k, Aq], float)/Aq;  mci2t = np.ones([k, Aq], float)/Aq;
    mci1t[map_inds_c1, :] = mci[::2]
    mci2t[map_inds_c2, :] = mci[1::2]
    
    if self.track_messages == 2:
      self.bi[:,:] = mbi
      self.c1i[:,:] = mci1t
      self.c2i[:,:] = mci2t

    mci1 = mci1t.flatten()
    mci2 = mci2t.flatten()
    in1 = mbi.flatten()
    tu1 = 0 # skip update C1 
    tu2 = 0 # skip update C2
    o1_test = np.zeros(np.shape(in1))
    o2_test = np.zeros(np.shape(in1))
    for i in range(0, K):
      if not tu1:
        out1 =  self.code.update(mci1, in1)
        oo = out1.reshape(k, Aq)
        in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
        in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
        if np.abs(out1-o1_test).sum() < thr: # if the update does not differ to much
          tu1 = 1
        else:
          o1_test[:] = out1
      if not tu2:
        out2t = self.code.update(mci2, in2)
        out2 = out2t.reshape(k, Aq)[deinter]
        in1 = (out2 * mbi)
        in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(Aq)))
        if np.abs(out2t-o2_test).sum() < thr: # if the update does not differ to much
          tu2 = 1
        else:
          o2_test[:] = out2t
      if self.track_messages > 0:
        self.i1o2[:, :, i] = in1.reshape(k, Aq) # input to dec 1 (out dec 2)
        self.i2o1[:, :, i] = in2.reshape(k, Aq) # input to dec 2 (out dec 1)

    mco1, out1 = self.code.update_both(mci1, in1)
    oo = out1.reshape(k, Aq)
#    in2 = (out1 * mci1).reshape(N, 4)[inter]
    in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
    mco2, out2t = self.code.update_both(mci2, in2)
    out2 = out2t.reshape(k, Aq)[deinter].flatten()

    out = (out1 * out2 * mbi.flatten()).reshape(k, Aq)
#  print out1.reshape(k, 2),  out2.reshape(k, 2)
    mbo = (out.flatten()/ (out.sum(axis=1).repeat(Aq))).reshape(k, Aq)
    mco = np.zeros([p, Aq], float)
    mco[::2, :] = mco1.reshape(k, Aq)[map_inds_c1, :]
    mco[1::2, :] = mco2.reshape(k, Aq)[map_inds_c2, :]
    if self.track_messages == 2:
      self.bo[:,:] = mbo
      self.c1o[:,:] = mco1.reshape(k, Aq)
      self.c2o[:,:] = mco2.reshape(k, Aq)
    return np.vstack([mbo, mco])
 

  def decode_ef(self, mu_i):
    '''
      stream decode - optimized - soft out\
      mu_i .... input metric Assumed to be float32 type
    '''
    if self.eff != 32:
      print 'Please enable eff flag'
    K = self.K
    k = np.uint32(self.k)
    p = self.p
    Aq = self.Aq
    inter = self.inter
    deinter = self.deinter
    mbi = mu_i[:k,:]
    mci = mu_i[k:,:]
    map_inds_c1 = self.map_inds_c1
    map_inds_c2 = self.map_inds_c2

    mci1t = np.ones([k, Aq], np.float32)/Aq;  mci2t = np.ones([k, Aq], np.float32)/Aq;
    mci1t[map_inds_c1, :] = mci[::2]
    mci2t[map_inds_c2, :] = mci[1::2]

#    print 'mbi:\n', mbi
#    print 'mci1t\n', mci1t
#    print 'mci2t\n', mci2t

    mci1 = mci1t.flatten()
    mci2 = mci2t.flatten()
    in1 = mbi.flatten()

    _Turbo_lib.stream_turbo_decode_funct(in1, mci1, mci2, self.S, self.Q, \
                                         self.inter, self.deinter, np.uint32(K))
    
    mco = np.zeros([p, Aq], np.float32)
    mco[::2, :] = mci1.reshape(k, Aq)[map_inds_c1, :]
    mco[1::2, :] = mci2.reshape(k, Aq)[map_inds_c2, :]

    mbo = in1.reshape(k, Aq)

    return np.vstack([mbo, mco])
 


  def create_code(self, Aq = 2, depth = 3):
    '''
    Aq-ary alphabet with given depth. Number of modulator states (and corresponding complexity) is
    given by Aq ** (depth).

    TODO -> add octal generator
    '''
    if depth == 1:
      S = np.array([[0,1],[1,0]], int)
      Q = np.array([[0,1],[1,0]], int)
    if depth == 3:
      S = np.zeros([Aq,Aq**3],int)
      Q = np.zeros([Aq,Aq**3], int)
      for s in range(Aq**3):
        for d in range(Aq):
          s0 = s % Aq
          s1 = (s / Aq) % Aq
          s2 = (s / Aq**2) % Aq
          S[d,s] = Aq**2 * (s2 ^ d) + Aq * (s0 ^ s2) + s1
          Q[d,s] = s2 ^ s0 ^ d
#      _Turbo_lib.create_matrix_SQ_XOR_mod(S, Q)    
      _Turbo_lib.create_matrix_SQ_XOR(S, Q)    
    return FSM(S, Q)

  def print_info(self):
    print 'Turbo code information:\nk ... %d\nn ... %d\nrate ... %.3f\n'%(self.k, self.n, self.rate)

  def test(self, SNR = 0):
    '''
    BPSK in AWGN test of the turbo encoder
    '''
    b = np.random.randint(2, size=self.k)
    c = self.encode(b)
    s = np.array([-1,1])[c] # BPSK mapper

    ## AWGN ##
    alpha = 10**(float(SNR)/10) 
    sigma2w = 1.0/alpha
    real_noise = np.random.normal(0,np.sqrt(sigma2w),np.shape(s/2))
    imag_noise = 1j * np.random.normal(0,np.sqrt(sigma2w),np.shape(s/2))
    noise = 1.0/np.sqrt(2)  * (real_noise + imag_noise)
    x = s + noise

    ## Demodulator ##
    lenX, = x.shape
    lenC = 2
    if self.eff == 32:
      mu = np.zeros([lenX, lenC], np.float32)
    else:
      mu = np.zeros([lenX, lenC], np.float)
    for i in range(0, lenC):
      mu[:, i] = np.exp(-np.abs(x - np.array([-1,1])[i])**2/sigma2w)
    if (np.min(mu) == 0):
      mu += 1e-100   # Avoiding all-zero vector
    mu = mu / np.sum(mu,axis = 1).repeat(lenC).reshape(lenX, lenC)
    
    if self.eff == 32:
      mu_o = self.decode_ef(mu)
    else: 
      mu_o = self.decode(mu)
    
    if self.track_messages == 2:
      self.b = b
      self.c = c
      self.s = s
      self.x = x
      self.mu = mu
      self.mu_o = mu_o

#      self.err_eval()
    print 'nerr_uncoded:', np.sum(np.argmax(mu[:self.k,:], axis = 1) != b )
    print 'nerr_coded:',   np.sum(np.argmax(mu_o[:self.k,:], axis = 1) != b )
    return b, mu_o
  
  def err_eval(self, mu, mu_o, b):
    print 'nerr_uncoded:', np.sum(np.argmax(mu[:len(b),:], axis = 1) != b )
    print 'nerr_coded:',   np.sum(np.argmax(mu_o[:len(b),:], axis = 1) != b )

  def draw(self):
    ''' 
    The function drawing FN corresponding to given factor Graph
    '''
    cd = self.k + 3  # distance of most left data and the coder
    ## C1 -- nodes
    B1 = [node(i, -i*2, i, '$b_{'+str(i)+'}$', 'VN') for i in range(self.k)] # data VNs
    S1 = [node(cd, -(-1 + 2*i), self.k + i, '$s_{'+str(i)+'}$', 'VN') for i in range(self.k + 1)] # state VNs
    C1 = [node(cd, -2*i, 2*self.k + 1 + i, '', 'FN') for i in range(self.k)] # state Factor Nodes
    Cc1 = [node(cd+3, -i*2,i + 3*self.k + 1, '$c_{1,'+str(i)+'}$', 'VN') for i in range(self.k)] # C1 VNs
    ## C1 -- connect edges
    E_BC1 = [edge(B1[i], C1[i], '', ['pos=0.5,above'], ['\\to'+create_latex_array(self.i1o2[i,:,:].T)]) for i in range(self.k)] # data to code
    E_SC1 = [edge(S1[i], C1[i], '') for i in range(self.k)] # s-1 to code
    E_CS1 = [edge(C1[i], S1[i+1], '') for i in range(self.k)] # code to s+1
    E_CC1c = [edge(C1[self.map_inds_c1[i]], Cc1[self.map_inds_c1[i]], '', \
    ['pos=0.5,above', 'pos=0.5,below', 'pos=1.3, above, text=red'], \
    ['\\leftarrow'+create_latex_array(self.c1i[self.map_inds_c1[i],:]), '\\to'+create_latex_array(self.c1o[self.map_inds_c1[i],:]), \
    self.c[self.k+2*i]]\
    ) for i in range(len(self.map_inds_c1))] # code to c_i
    E_CC1n = [edge(C1[i], Cc1[i], '', \
    ['pos=0.5,above', 'pos=0.5,below'], \
    ['\\leftarrow'+create_latex_array(self.c1i[i,:]), '\\to'+create_latex_array(self.c1o[i,:])]\
    ) for i in range(self.k) if i not in self.map_inds_c1] # code to c_i

    ## C2 -- nodes
    s2 = -2*self.k-2 # skip between C1 and C2
    st_id = 4*self.k + 1
    Bi = [node(self.inter[i], s2-i*2, i + st_id, '', 'dot') for i in range(self.k)] # data VNs
    S2 = [node(cd, s2-(-1 + 2*i), self.k + i + st_id, '$s_{'+str(i)+'}$', 'VN') for i in range(self.k + 1)] # state VNs
    C2 = [node(cd, s2-2*i, 2*self.k + 1 + i + st_id, '', 'FN') for i in range(self.k)] # state Factor Nodes
    Cc2 = [node(cd+3, s2-i*2,i + 3*self.k + 1 + st_id, '$c_{2,'+str(i)+'}$', 'VN') for i in range(self.k)] # C2 VNs
    ## C2 -- connect edges
    E_BBi = [edge(B1[self.inter[i]], Bi[i], '' ) for i in range(self.k)] # data to code
    E_BC2 = [edge(Bi[i], C2[i], '', ['above, fill=white, pos=0.5'],\
            ['\\to'+create_latex_array(self.i2o1[i,:,:].T)]) for i in range(self.k)] # data to code
    E_SC2 = [edge(S2[i], C2[i], '') for i in range(self.k)] # s-1 to code
    E_CS2 = [edge(C2[i], S2[i+1], '') for i in range(self.k)] # code to s+1
    E_CC2c = [edge(C2[self.map_inds_c2[i]], Cc2[self.map_inds_c2[i]], '', \
    ['pos=0.5,above', 'pos=0.5,below', 'pos=1.3, above, text=red'], \
    ['\\leftarrow'+create_latex_array(self.c2i[self.map_inds_c2[i],:]), '\\to'+create_latex_array(self.c2o[self.map_inds_c2[i],:]), \
    self.c[self.k+1+2*i]]\
    ) for i  in range(len(self.map_inds_c2))] # code to c_i
    E_CC2n = [edge(C2[i], Cc2[i], '', \
    ['pos=0.5,above', 'pos=0.5,below'], \
    ['\\leftarrow'+create_latex_array(self.c2i[i,:]), '\\to'+create_latex_array(self.c2o[i,:])]\
    ) for i in range(self.k) if i not in self.map_inds_c2] # code to c_i
#    E_CC2 = [edge(C2[i], Cc2[i], '') for i in range(self.k)] # code to c_i
    
    st_id = 8*self.k + 2
    SS = [node(-3, -i*2, i + st_id, '', 'dot') for i in range(self.k)] # input data source
    E_SSB = [edge(B1[i], SS[i], '', \
    ['pos=0.5,above,fill=white', 'pos=0.5,below, fill=white', 'pos=1, above, color=red'], \
    ['\\leftarrow'+create_latex_array(self.bo[i,:]), '\\to'+create_latex_array(self.bi[i,:]), str(self.b[i])]\
    ) for i in range(self.k)] # input to data
#    E_SSB = [edge(B1[i], SS[i], '') for i in range(self.k)] # input to data
    ## -- starting input
    
    f = open('text/d.tex','w')
    f.write('''
\\begin{tikzpicture}[
FN/.style={rectangle,inner sep=0pt,minimum size=2ex, draw=black},
VN/.style={circle,inner sep=0pt,minimum size=2ex, draw=black},
dot/.style={circle,inner sep=0pt,minimum size=5pt, draw=black, fill=black},
]''')
    for i in [B1, S1, C1, Cc1, Bi, S2, C2, Cc2, SS,\
              E_BC1, E_SC1, E_CS1, E_CC1c, E_CC1n, E_BBi, E_BC2, E_SC2, E_CS2, E_CC2c, E_CC2n, E_SSB]:
      for a in i:
        f.write(a.draw())
    f.write('\\end{tikzpicture}')
      
    f.close()

def create_latex_array(matrix):
  if len(matrix.shape) == 2: # matrix
    x,y = matrix.shape
    out = '\\left[\\begin{array}{'+''.join(['c' for i in range (y)])+'}\n'
    for i in range(x):
      for j in range(y-1):
        out = out+str(round(matrix[i,j], 4))+'&'
      out = out+str(round(matrix[i,y-1], 4))+'\\\\\n'
    out = out+'\\end{array}\\right]'
    return out
  elif len(matrix.shape) == 1: # vector
    x = len(matrix)
    out = '\\left[\\begin{array}{'+''.join(['c' for i in range (x)])+'}\n'
    for i in range(x-1):
      out = out+str(round(matrix[i], 4))+'&'
    out = out+str(round(matrix[x-1], 4))+'\n\\end{array}\\right]'
    return out
  else:
    return ''



class node():
  ''' 
  For tikz drawing
  '''
  def __init__(self, x, y, i, label, style):
    self.x = x # x position 
    self.y = y # y position
    self.i = i # id
    self.label = label # label
    self.style = style 

  def draw(self):
    return '\\node[%s] at (%.4f,%.4f) (n%d) {%s};\n' %(self.style, self.x, self.y, self.i, self.label)


class edge():
  '''
  edge between nodes n1 and n2
  '''
  def __init__(self, n1, n2, style, node_desr=[], node_str=[]):
    self.n1 = n1.i
    self.n2 = n2.i
    self.style = style
    self.node_desr = node_desr
    self.node_str = node_str
  
  def draw(self):
    if self.node_desr == []:
      return '\\draw[%s] (n%d) -- (n%d);\n' %(self.style, self.n1, self.n2)
    else: 
      app = ''
      for i in range(len(self.node_desr)):
        descr = self.node_desr[i]
        lab = self.node_str[i]
        app = app+'node[%s] {\\tiny{$%s$}} '%(descr, lab)
      return '\\draw[%s] (n%d) -- (n%d)'%(self.style, self.n1, self.n2)+app+';\n'


def expand_matrices(S, Q): # expand the matrix from GFq to GF(q**2)
  [Md, Ms] = S.shape # Md supposed to equal 2
#  res = np.zeros([Md ** n, Ms ** n],int)
#  print [[(np.mod(i, Md), np.mod(j, Ms), np.mod(i, Md**2)/Md, np.mod(j, Ms**2)/Ms)\
#  for i in np.arange(Md**2)] for j in np.arange(Ms**2)]
  SE = np.asarray([[\
  S[np.mod(i, Md), np.mod(j, Ms)]+Ms* S[np.mod(i, Md**2)/Md, np.mod(j, Ms**2)/Ms]
  for j in np.arange(Ms**2)] for i in np.arange(Md**2)])
  QE = np.asarray([[\
  Q[np.mod(i, Md), np.mod(j, Ms)]+Md* Q[np.mod(i, Md**2)/Md, np.mod(j, Ms**2)/Ms]
  for j in np.arange(Ms**2)] for i in np.arange(Md**2)])
  return SE, QE

#@jit
#def update_FSM(N, pc, pd):  # Fixed for the simplest recursive FSM
#  ps1 = np.ones(N + 1) * 0.5
#  ps = np.ones(N + 1) * 0.5
#  ps[0] = 1
#  out = np.zeros(N)
#  for i in np.arange(0, N): # Forward
#    p0 = ps[i] * pc[i] * pd[i] + (1-ps[i])*pc[i]*(1-pd[i])
#    p1 = ps[i]*(1-pc[i])*(1-pd[i]) + (1-ps[i]) * (1-pc[i]) * (pd[i]) 
#    ps[i + 1] = p0 / (p0 + p1) 
#  for i in np.arange(N-1, 0, -1): # Backward
#    p0 = ps1[i+1] * pc[i] * pd[i] + (1-ps1[i+1])*(1-pc[i])*(1-pd[i])
#    p1 = (1-ps1[i+1])*(1-pc[i])*(pd[i]) + ps1[i+1] * pc[i] * (1-pd[i]) 
#    ps1[i] = p0 / (p0 + p1) 
#  for i in range(0, N): # Data
#    p0 = ps[i] * ps1[i+1] * pc[i] + (1-ps[i]) * (1-ps1[i+1]) * (1-pc[i])
#    p1 = ps[i] * (1-ps1[i+1]) * (1-pc[i]) + (1-ps[i]) * ps1[i+1] * pc[i]
#    out[i] = p0 / (p0 + p1) 
#  return out

def decode_only_CC(N, p0, code, d): # decode the simplest turbo code
  N = len(d)
  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  
  in1 = p0_syst
  out = np.zeros(len(in1));
  _Turbo_lib.update_FSM_func(p0_first, p0_syst, out)
  
  res = (out * p0_syst) / (1 - out - p0_syst + 2* (out * p0_syst))
  return np.sum(np.abs(d-(res < 0.5)))/float(N)

def decode_simple(p0, code, d, inter, deinter, K): # decode the simplest turbo code
  N = len(d)
  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  p0_sec = p0[(2*N):]
  
  in1 = p0_syst
  out1 = np.zeros(N)
  out2t = np.zeros(N)

  for i in range(0, K):
    _Turbo_lib.update_FSM_func(p0_first, in1, out1)
    in2 = ((out1 * p0_syst) / (1 - out1 - p0_syst + 2* (out1 * p0_syst)))[inter]
    _Turbo_lib.update_FSM_func(p0_sec, in2, out2t)
    out2 = out2t[deinter]
    in1 = ((out2 * p0_syst) / (1 - out2 - p0_syst + 2* (out2 * p0_syst)))    
  
  res = out2*out1*p0_syst / (1 - out2 - out1 - p0_syst + \
  out2 * out1 + out1 * p0_syst + out2 * p0_syst)
  return np.sum(np.abs(d-(res < 0.5)))/float(N)

def eval_EXIT_FSM(num_EXIT, pc, p0_syst, d):
  LLR_in = generate_samples(d, num_EXIT) # A
  pd = 1. / (1 + np.exp(LLR_in))# pr(d = 0)
  out = np.zeros(pd.shape,dtype=float)
#  LLR_chan = np.log((1-pc)/pc)  # Z
  for i in range(0, num_EXIT):
    act_out = np.zeros(pc.shape, dtype=float) # 
#    out1 = np.zeros(act_out.shape,dtype=float)
    pin = ((pd[i,:] * p0_syst) / (1 - pd[i,:] - p0_syst + 2* (pd[i,:] * p0_syst)))
    _Turbo_lib.update_FSM_func(pc, pin, act_out)
#    _Turbo_lib.update_general_FSM_func(pc, pin, out1,np.array([0,1,1,0]), np.array([0,1,1,0]))
#    print out1 - act_out
#    _Turbo_lib.update_FSM_func(pc, pd[i,:], act_out)
#    ((act_out1 * p0_syst) / (1 - out1 - p0_syst + 2* (out1 * p0_syst)))
    out[i, :] = np.log((1-act_out)/(act_out))
  MI = evaluate_extrinsic_info_NI(out, d.flatten())
#  print evaluate_extrinsic_info_NI(LLR_in, d.flatten())
  return MI
  
def eval_EXIT_FSM_GF4(num_EXIT, pc, p0_syst, dA, dB):
  d = dA ^ dB
  LLR_in = generate_samples(d, num_EXIT) # A
  p0h = 1./(1+np.exp(LLR_in))
  p1h = np.exp(LLR_in)/(1+np.exp(LLR_in))

# Zero additional knowledge
  p0 = p0h / 2
  p3 = p0h / 2
  p1 = p1h / 2
  p2 = p1h / 2

# Full additional knowledge
  index = dA * 2 + dB
  ind0 = np.nonzero(index==0)[0]
  ind1 = np.nonzero(index==1)[0]
  ind2 = np.nonzero(index==2)[0]
  ind3 = np.nonzero(index==3)[0]
#  p0[:,ind0] = p0h[:,ind0] 
#  p3[:,ind3] = p0h[:,ind3] 
#  p0[:,ind3] = 0 
#  p3[:,ind0] = 0 
#  p1[:,ind1] = p1h[:,ind1] 
#  p2[:,ind2] = p1h[:,ind2]
#  p1[:,ind2] = 0 
#  p2[:,ind1] = 0

# proportional knowledge
  indh0 = np.union1d(ind0,ind3) # c = 0 or 3 
  indh1 = np.union1d(ind1,ind2) # c = 1 or 2
  cch = np.zeros([num_EXIT, len(d)])
  cch[:, ind3] = 1 # c = 3 -> cch = 1, c = 0 -> cch = 0
  cch[:, ind2] = 1 # c = 2 -> cch = 1, c = 1 -> cch = 0
  LLRc = generate_samples(cch, num_EXIT)
  p0[:,indh0] = p0h[:, indh0] / (1 + np.exp(LLRc[:, indh0]))
  p3[:,indh0] = p0h[:, indh0] * np.exp(LLRc[:, indh0]) / (1 + np.exp(LLRc[:, indh0]))

#  p1[:,indh1] = p1h[:, indh1] / (1 + np.exp(LLRc[:, indh1]))
#  p2[:,indh1] = p1h[:, indh1] * np.exp(LLRc[:, indh1]) / (1 + np.exp(LLRc[:, indh1]))


  
  out = np.zeros([num_EXIT, len(d)],dtype=float)
#  LLR_chan = np.log((1-pc)/pc)  # Z
  for i in range(0, num_EXIT):
    pd = np.vstack([p0[i,:], p1[i,:], p2[i,:], p3[i,:]]).transpose()
    act_out = np.zeros(4*len(d), dtype=float) # 
    pin = p0_syst * pd
    pin = pin / np.array([pin.sum(axis=1)]).repeat(4,axis=0).transpose()
    _Turbo_lib.update_FSM_GF4_func(pc.flatten(), pin.flatten(), act_out)
    oo = act_out.reshape(len(d), 4)
    out[i, :] = np.log((oo[:,1] + oo[:,2])/(oo[:,0]+oo[:,3]))
  MI = evaluate_extrinsic_info_NI(out, d.flatten())
#  print evaluate_extrinsic_info_NI(LLR_in, d.flatten())
  return MI
  
def track_turbo_dec(p0, code, d, inter, deinter, K): # decode the simplest turbo code
  N = len(d)
  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  p0_sec = p0[(2*N):]
  
  in1 = p0_syst
  out1 = np.zeros(N)
  out2t = np.zeros(N)

  res_C1 = np.zeros([K + 1, N], float)
  res_C2 = np.zeros([K + 1, N], float)
#  res_C2[0,:] = np.log((1-in1)/in1)
#  plt.figure()
  for i in range(0, K):

    _Turbo_lib.update_FSM_func(p0_first, in1, out1)
    res_C2[i + 1, :] = np.log((1-out1)/out1)
#    res_C1[i + 1, :] = np.log((1-out1)/out1) - np.log((1-p0_syst)/p0_syst)
    in2 = ((out1 * p0_syst) / (1 - out1 - p0_syst + 2* (out1 * p0_syst)))
 #   res_C1[i + 1, :] = np.log((1-in2)/in2)
    in2 = in2[inter]

#    a = np.histogram(res_C2[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'-')

    _Turbo_lib.update_FSM_func(p0_sec, in2, out2t)
    out2 = out2t[deinter]
    res_C1[i + 1, :] = np.log((1-out2)/out2)
#    res_C2[i + 1, :] = np.log((1-out2)/out2) - np.log((1-p0_syst)/p0_syst)
    in1 = ((out2 * p0_syst) / (1 - out2 - p0_syst + 2* (out2 * p0_syst)))    
#    res_C2[i + 1, :] = np.log((1-in1)/in1)

#    a = np.histogram(res_C1[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'--')
  
#  plt.show()

  MI_C1 = evaluate_extrinsic_info_NI(res_C1, d.flatten())
  MI_C2 = evaluate_extrinsic_info_NI(res_C2, d.flatten())

  res = out2*out1*p0_syst / (1 - out2 - out1 - p0_syst + \
  out2 * out1 + out1 * p0_syst + out2 * p0_syst)
  print np.sum(np.abs(d-(res < 0.5)))/float(N)
  return np.array([MI_C1,MI_C2])

def track_turbo_dec_HDF_GF4(mu, d,  inter, deinter, K):
  N = len(d)
  mu_syst = mu.transpose()[:,0:N].transpose().flatten() 
  mu_first = mu.transpose()[:,N:(2*N)].transpose().flatten()
  mu_sec = mu.transpose()[:,(2*N):].transpose().flatten()
  
  in1 = mu_syst
  out1 = np.zeros(4*N)
  out2t = np.zeros(4*N)

  res_C1 = np.zeros([K + 1, N], float)
  res_C2 = np.zeros([K + 1, N], float)
#  plt.figure()
  for i in range(0, K):
    _Turbo_lib.update_FSM_GF4_func(mu_first, in1, out1)
    oo = out1.reshape(N, 4)
    res_C2[i + 1, :] = np.log((oo[:,1] + oo[:,2])/(oo[:,0] + oo[:,3]))
    in2 = (out1 * mu_syst).reshape(N, 4)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(4)))

#    a = np.histogram(res_C2[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'-')

    _Turbo_lib.update_FSM_GF4_func(mu_sec, in2, out2t)
    out2 = out2t.reshape(N, 4)[deinter]
    res_C1[i + 1, :] = np.log((out2[:,1]+out2[:,2])/(out2[:,0]+out2[:,3]))
    in1 = (out2 * (mu_syst.reshape(N, 4)))
    in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(4)))

#    a = np.histogram(res_C1[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'--')
#  plt.show()
  
  MI_C1 = evaluate_extrinsic_info_NI(res_C1, d.flatten())
  MI_C2 = evaluate_extrinsic_info_NI(res_C2, d.flatten())

  res = out2*((out1*mu_syst).reshape(N, 4))
  res = res/res.sum(axis=1).repeat(4).reshape(N,4)
  est = res[:,0] + res[:,3] < res[:,2] + res[:,1]

  print np.sum(d != est)/float(N)
  return np.array([MI_C1,MI_C2])




def P2P_EXIT_Analysis(SNR, s, const, K, d, code, inter, deinter):
  N = len(d)
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in const])
  p0 = mu[0,:] / mu.sum(axis=0) 
  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  MI = eval_EXIT_FSM(21, p0_first, p0_syst, d)
  XX = track_turbo_dec(p0, code, d, inter, deinter, K) 
  return [MI, XX]
#  return MI


def run_P2P(SNR, s, const, K, d, code, inter, deinter):
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in const])
  p0 = mu[0,:] / mu.sum(axis=0) 
  return decode_simple(p0, code, d, inter, deinter, K)

def run_P2P_CC(SNR, s, const, K, d, N, code):
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in const])
  p0 = mu[0,:] / mu.sum(axis=0) 
  return decode_only_CC(N, p0, code, d) 

def run_uncoded(SNR, s, const, K, d, N):
  (x,sigma2w) = AWGN(s[0:N], SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in const])
  return np.sum(np.abs(np.argmax(mu,axis=0)-d))/float(N)

def run_BPSK_system(N, SNR = -1, K=20):
#  code = FSM(np.array([[0,1],[1,0]]), np.array([[0,2],[3,1]]), 1,2,2,2)
#  code_N = FSM(np.array([[0,0,1,1],[2,2,3,3]]), np.array([[0,1,1,0],[1,0,0,1]]), 1,2,2,2)

  code = FSM(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
  d = np.random.randint(2,size=N)

  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i
  
  d_int = d[inter]
  c_N = code.encode(d, 0)
  c_R = code.encode(d_int, 0)
#  print 'systematic:\t',d,'\nencoded:\t',c_N,'\ninter_enc:\t', c_R
  
#  SNR_vals = np.linspace(-4,6,16,True)
#  SNR = -2
  
  BPSK = np.array([-1.,1.])
  s = BPSK[np.hstack([d, c_N, c_R])]
#  K = 20 # Number of iterations for decoder
  A = P2P_EXIT_Analysis(SNR, s, BPSK, K, d, code, inter, deinter)
  plot_dec(A[1])
  plot_EXIT(A[0],A[0])
#  plt.show()
#  run_QPSK(N, SNR, K)

#  Pbe = np.asarray([run_P2P(SNR, s, BPSK, K,d, code, inter, deinter) for SNR in SNR_vals])

#  Pbe_cod = Parallel(n_jobs=8)(delayed(run_P2P)(float(SNR),s,BPSK,K, d,code, inter, deinter) for SNR in SNR_vals)
#  Pbe_unc = Parallel(n_jobs=8)(delayed(run_uncoded)(float(SNR),s,BPSK,K, d, N) for SNR in SNR_vals)
#  Pbe_codCC = Parallel(n_jobs=8)(delayed(run_P2P_CC)(float(SNR),s,BPSK,K, d, N,code) for SNR in SNR_vals)
 

  plot = 0
  if plot:
    plt.semilogy(SNR_vals, np.asarray(Pbe_cod) , 'x-k', ms=10,lw=2)
    plt.semilogy(SNR_vals, np.asarray(Pbe_unc) , 'p-b', ms=10,lw=2)
    plt.semilogy(SNR_vals, np.asarray(Pbe_codCC) , 'h-r', ms=10,lw=2)
    plt.legend(['Turbo Coded (r=1/3)', 'Uncoded (r=1)', 'Convolutionaly Coded (r=1/2)'])
    plt.grid()
    plt.show()
  return A


def run_QPSK(N, SNR, K = 20):
  plus = np.asarray([[i ^ j for i in range(0,4)] for j in range(0,4)])
  code = FSM(plus,plus)
  d = np.random.randint(4,size=N)
  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i
  
  d_int = d[inter]
  c_N = code.encode(d, 0)
  c_R = code.encode(d_int, 0)
  
  QPSK = np.array([-1.,1j, 1., -1j])
  s = QPSK[np.hstack([d, c_N, c_R])]

#  Pbe = np.asarray([run_P2P(SNR, s, BPSK, K) for SNR in SNR_vals])

  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in QPSK])
  mu = mu / mu.sum(axis=0) 

  mu_syst = mu[:,0:N].transpose().flatten() 
  mu_first = mu[:,N:(2*N)].transpose().flatten()
  mu_sec = mu[:,(2*N):].transpose().flatten()
  
  in1 = mu_syst
  out1 = np.zeros(4*N)
  out2t = np.zeros(4*N)

  for i in range(0, K):
    _Turbo_lib.update_FSM_GF4_func(mu_first, in1, out1)
    in2 = (out1 * mu_syst).reshape(N, 4)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(4)))

    _Turbo_lib.update_FSM_GF4_func(mu_sec, in2, out2t)
    out2 = out2t.reshape(N, 4)[deinter]
    in1 = (out2 * (mu_syst.reshape(N, 4)))
    in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(4)))
#    print np.isnan(out2).sum()
  
  res = out2*((out1*mu_syst).reshape(N, 4))
  res = res/res.sum(axis=1).repeat(4).reshape(N,4)
  est = np.argmax(res,axis=1)

  return np.sum(d != est)/float(N)

def run_qPSK(N, SNR, K = 20, q=8):
  plus = np.asarray([[i ^ j for i in range(0,q)] for j in range(0,q)])
  code = FSM(plus,plus)
  d = np.random.randint(q,size=N)
  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i
  
  d_int = d[inter]
  c_N = code.encode(d, 0)
  c_R = code.encode(d_int, 0)
  
  PSK8 = np.exp(-1j*np.pi/q*2*np.arange(q))
  s = PSK8[np.hstack([d, c_N, c_R])]

#  Pbe = np.asarray([run_P2P(SNR, s, BPSK, K) for SNR in SNR_vals])

  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in PSK8])
  mu = mu / mu.sum(axis=0) 

  mu_syst = mu[:,0:N].transpose().flatten() 
  mu_first = mu[:,N:(2*N)].transpose().flatten()
  mu_sec = mu[:,(2*N):].transpose().flatten()
  
  in1 = mu_syst
  out1 = np.zeros(q*N)
  out2t = np.zeros(q*N)

  for i in range(0, K):
    _Turbo_lib.update_FSM_GFq_func(mu_first, in1, out1,q)
    in2 = (out1 * mu_syst).reshape(N, q)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(q)))

    _Turbo_lib.update_FSM_GFq_func(mu_sec, in2, out2t, q)
    out2 = out2t.reshape(N, q)[deinter]
    in1 = (out2 * (mu_syst.reshape(N, q)))
    in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(q)))
#    print np.isnan(out2).sum()
  
  res = out2*((out1*mu_syst).reshape(N, q))
  res = res/res.sum(axis=1).repeat(q).reshape(N,q)
  est = np.argmax(res,axis=1)

  return np.sum(d != est)/float(N)

def run_HDF(N, SNR, K):
  code = FSM(np.array([[0,1],[1,0]]), np.array([[0,1],[1,0]]))
  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i
  
  bA = np.random.randint(2, size=N)
  bA_int = bA[inter]
  cA_N = code.encode(bA, 0)
  cA_R = code.encode(bA_int, 0)
  cA = np.hstack([bA, cA_N, cA_R])
  
  bB = np.random.randint(2, size=N)
  bB_int = bB[inter]
  cB_N = code.encode(bB, 0)
  cB_R = code.encode(bB_int, 0)
  cB = np.hstack([bB, cB_N, cB_R])
  
  hA = 1.
  hB = 1.
  BPSK = np.array([-1.,1.])
  rConst = np.asarray([hA * a + hB * b for a in BPSK for b in BPSK])
  
  s = hA*BPSK[cA] + hB*BPSK[cB]
  (x,sigma2w) = AWGN(s, SNR)
  x = x.flatten()
  muR = np.zeros([N * 3 , 4], float) # 4ARY constellation fixed
  
  #for i in range(0,N * M):
  #  muR[i,:] = np.asarray([np.exp(-np.abs(x[i] - s0)**2/sigma2w) for s0 in rConst]).flatten()
  for i in range(0,len(rConst)):
    muR[:,i] = np.exp(-np.abs(x - rConst[i])**2/sigma2w)
  
  suma = muR.sum(axis = 1)
  for i in range(0,len(rConst)):
    muR[:,i] = muR[:,i] / suma

  mu = np.zeros([N * 3, 2], order='F')
  
  mu[:,0] = muR[:,0] + muR[:,3]
  mu[:,1] = muR[:,1] + muR[:,2]

  MI = eval_EXIT_FSM(21, mu[N:(2*N),0], mu[:N,0], bA^bB)
  XX = track_turbo_dec(mu[:,0], code, bA^bB, inter, deinter, K) 
  XX1 = track_turbo_dec_HDF_GF4(muR, bA^bB, inter, deinter, K)
  MI1 = eval_EXIT_FSM_GF4(21, muR[N:(2*N),:], muR[:N,:], bA, bB)
#  print decode_simple(mu[:,0], code, bA^bB, inter, deinter, K)
  plot_dec(XX, 'ro-', 5, 1)
  plot_dec(XX1, 'bp-', 5, 1)
  plot_EXIT(MI,MI, ['r--x', 'r--o'], [5,5],[1,1])
  plot_EXIT(MI1,MI1, ['b--x', 'b--o'], [5,5],[1,1])
#  plt.show()
  return [MI, XX, MI1, XX1] 

def test_general_code(N, SNR = -1, K=20):
#  code = FSM(np.array([[0,1],[1,0]]), np.array([[0,2],[3,1]]), 1,2,2,2)
#  code_N = FSM(np.array([[0,0,1,1],[2,2,3,3]]), np.array([[0,1,1,0],[1,0,0,1]]), 1,2,2,2)

  S = np.zeros([2,16],int)
  Q = np.zeros([2,16], int)
  for s in range(16):
    for d in range(2):
        [s0, s1, s2,s3] = dec2bin(s,4)
        S[d,s] = bin2dec(np.array([s3^d, s2^s0,s1,s2]))
        Q[d,s] = d^s0^s1

#  S = np.array([[0,2,1,2],[1,3,1,3]])
#  Q = np.array([[0,1,1,0],[1,0,0,1]])
#  S = np.array([[0,1],[1,0]])
#  Q = np.array([[0,1],[1,0]])
  code = FSM(S,Q)
  d = np.random.randint(2,size=N)

  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i
  
  d_int = d[inter]
  c_N = code.encode(d, 0)
  c_R = code.encode(d_int, 0)
  BPSK = np.array([-1.,1.])
  s = BPSK[np.hstack([d, c_N, c_R])]
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in BPSK])
  p0 = mu[0,:] / mu.sum(axis=0) 
  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  p0_first[::2] = 0.5 # puncturing
  p0_first[1::16] = 0.5 # puncturing
  p0_sec = p0[(N*2):]
  p0_sec[::2] = 0.5 # puncturing 
  p0_sec[1::16] = 0.5 # puncturing
  num_EXIT = 21
  LLR_in = generate_samples(d, num_EXIT) # A
  pd = 1. / (1 + np.exp(LLR_in))# pr(d = 0)
  out = np.zeros(pd.shape,dtype=float)
#  LLR_chan = np.log((1-pc)/pc)  # Z

# Evaluate EXIT curves
  for i in range(0, num_EXIT):
#    act_out = np.zeros(2*N, dtype=float) # 
    pin = ((pd[i,:] * p0_syst) / (1 - pd[i,:] - p0_syst + 2* (pd[i,:] * p0_syst)))    
    pinn = np.vstack([pin, 1-pin]).flatten(1)
    pc = np.vstack([p0_first, 1-p0_first]).flatten(1)
 #   _Turbo_lib.update_general_FSM_func(pc, pinn, act_out,S, Q)
    act_out = code.update(pc, pinn)
    out[i, :] = np.log(act_out[1::2]/act_out[::2])
  MI = evaluate_extrinsic_info_NI(out, d.flatten())

# Track the decoder run
  in1 = p0_syst
  res_C1 = np.zeros([K + 1, N], float)
  res_C2 = np.zeros([K + 1, N], float)
  for i in range(0, K):
    pc = np.vstack([p0_first, 1-p0_first]).flatten(1)
    pd = np.vstack([in1, 1-in1]).flatten(1)
#    act_out = np.zeros(2*N, dtype=float) # 
#    _Turbo_lib.update_general_FSM_func(pc, pd, act_out,S, Q)
    act_out = code.update(pc, pd)
    res_C2[i + 1, :] = np.log(act_out[1::2]/act_out[::2])
    out1 = act_out[::2]
#    res_C1[i + 1, :] = np.log((1-out1)/out1) - np.log((1-p0_syst)/p0_syst)
    in2 = ((out1 * p0_syst) / (1 - out1 - p0_syst + 2* (out1 * p0_syst)))
 #   res_C1[i + 1, :] = np.log((1-in2)/in2)
    in2 = in2[inter]

#    a = np.histogram(res_C2[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'-')

    pc = np.vstack([p0_sec, 1-p0_sec]).flatten(1)
    pd = np.vstack([in2, 1-in2]).flatten(1)
#    act_out = np.zeros(2*N, dtype=float) # 
#    _Turbo_lib.update_general_FSM_func(pc, pd, act_out,S, Q)
    act_out = code.update(pc, pd)
    out2t = act_out[::2]
    out2 = out2t[deinter]
    res_C1[i + 1, :] = np.log((1-out2)/out2)
#    res_C2[i + 1, :] = np.log((1-out2)/out2) - np.log((1-p0_syst)/p0_syst)
    in1 = ((out2 * p0_syst) / (1 - out2 - p0_syst + 2* (out2 * p0_syst)))    
#    res_C2[i + 1, :] = np.log((1-in1)/in1)

#    a = np.histogram(res_C1[i + 1, :], bins=50)
#    plt.plot(a[1][1:],a[0],'--')
  
#  plt.show()

  MI_C1 = evaluate_extrinsic_info_NI(res_C1, d.flatten())
  MI_C2 = evaluate_extrinsic_info_NI(res_C2, d.flatten())

  res = out2*out1*p0_syst / (1 - out2 - out1 - p0_syst + \
  out2 * out1 + out1 * p0_syst + out2 * p0_syst)
  print np.sum(np.abs(d-(res < 0.5)))/float(N)
 
  XX = np.asarray([MI_C1,MI_C2])
  plot_dec(XX, 'ko-', 5, 1)
  plot_EXIT(MI,MI, ['k--x', 'k--o'], [5,5],[1,1])
  plt.show()

  return [MI, XX]

def test_general_code_GF4(N, SNR = 5., K=20):
#  S = np.array([[0,2,1,2],[1,3,1,3]])
#  Q = np.array([[0,1,1,0],[1,0,0,1]])
#  S = np.array([[0,1],[1,0]])
#  Q = np.array([[0,1],[1,0]])
  S = np.zeros([2,16],int)
  Q = np.zeros([2,16], int)
  for s in range(16):
    for d in range(2):
        [s0, s1, s2,s3] = dec2bin(s,4)
        S[d,s] = bin2dec(np.array([s3^d, s2^s0,s1,s2]))
        Q[d,s] = d^s0^s1

  (SE, QE) = expand_matrices(S,Q)
  codeE = FSM(SE, QE)
  d = np.random.randint(4,size=N)
  c = codeE.encode(d,0)

  QPSK = np.array([-1.,1j, 1., -1j])
  s = QPSK[c]
  
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in QPSK])
  mu = mu / (mu.sum(axis=0))

  inp = np.ones(N * 4,float)/4 # no aprior info
  out = codeE.update(mu.flatten(1), inp).reshape(N,4)
  d_est = np.argmax(out,axis=1)
  nerr = np.sum(d_est != d)
  return nerr

def test_speed_decode(N = 10000, K = 20): 
  '''
    Testing FSM decode implementation
  '''
  Aq = 2
  SNR = -3
  S = np.zeros([Aq,Aq**3],int)
  Q = np.zeros([Aq,Aq**3], int)
  for s in range(Aq**3):
    for d in range(Aq):
      s0 = s % Aq
      s1 = (s / Aq) % Aq
      s2 = (s / Aq**2) % Aq
      S[d,s] = Aq**2 * (s2 ^ d) + Aq * (s0 ^ s2) + s1
      Q[d,s] = s2 ^ s0 ^ d

  
  code = FSM(S, Q)
  d = np.random.randint(2,size=N)

  inter = np.random.permutation(N)
  deinter = np.zeros(N, int)
  for i in range(0, N):
    deinter[inter[i]] = i

  d_int = d[inter]
  c_N = code.encode(d, 0)
  c_R = code.encode(d_int, 0)

  BPSK = np.array([-1.,1.])
  s = BPSK[np.hstack([d, c_N, c_R])]
  
  (x,sigma2w) = AWGN(s, SNR)
  mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in BPSK])
  p0 = mu[0,:] / mu.sum(axis=0)

  p0_syst = p0[0:N]
  p0_first = p0[N:(N*2)]
  p0_sec = p0[(2*N):]

  in1 = p0_syst
  out1 = np.zeros(N)
  out2t = np.zeros(N)

  _Turbo_lib.update_FSM_func(p0_first, in1, out1)
#  mco1, out1 = code.update_both(mci1, in1)
  out1 = code.update(p0_first, in1)
  mco1, out1 = update_both(self, p0_first, in1)


  return decode_simple(p0, code, d, inter, deinter, K)



def decode_eff_test(mu_i, K, k, p, Aq, inter, deinter, S, Q, map_inds_c1, map_inds_c2):
  '''
    Equal as Turbo_Code.decode_eff, but 15(?!) times faster
    stream decode - optimized - soft out\
    mu_i .... input metric Assumed to be float32 type
  '''
  mbi = mu_i[:k,:]
  mci = mu_i[k:,:]
  map_inds_c1 = map_inds_c1
  map_inds_c2 = map_inds_c2

  mci1t = np.ones([k, Aq], np.float32)/Aq;  mci2t = np.ones([k, Aq], np.float32)/Aq;
  mci1t[map_inds_c1, :] = mci[::2]
  mci2t[map_inds_c2, :] = mci[1::2]

  mci1 = mci1t.flatten()
  mci2 = mci2t.flatten()
  in1 = mbi.flatten()

  _Turbo_lib.stream_turbo_decode_funct(in1, mci1, mci2, S, Q, inter, deinter, K)

  mco = np.zeros([p, Aq], np.float32)
  mco[::2, :] = mci1.reshape(k, Aq)[map_inds_c1, :]
  mco[1::2, :] = mci2.reshape(k, Aq)[map_inds_c2, :]

  mbo = in1.reshape(k, Aq)
  return np.vstack([mbo, mco])

def decode_test(mu_i, K, k, p, Aq, inter, deinter, S, Q, map_inds_c1, map_inds_c2, code):
    '''
      stream decode - nonoptimized old fashioned - soft out\
      mu_i .... input metric
      K ... number of iterations in decoder
    '''
    mbi = mu_i[:k,:]
    mci = mu_i[k:,:]

    mci1t = np.ones([k, Aq], float)/Aq;  mci2t = np.ones([k, Aq], float)/Aq;
    mci1t[map_inds_c1, :] = mci[::2]
    mci2t[map_inds_c2, :] = mci[1::2]
    
    mci1 = mci1t.flatten()
    mci2 = mci2t.flatten()
    in1 = mbi.flatten()
    for i in range(0, K):
      out1 =  code.update(mci1, in1)
      oo = out1.reshape(k, Aq)
      in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
      in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
      out2t = code.update(mci2, in2)
      out2 = out2t.reshape(k, Aq)[deinter]
      in1 = (out2 * mbi)
      in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(Aq)))

    mco1, out1 = code.update_both(mci1, in1)
    oo = out1.reshape(k, Aq)
#    in2 = (out1 * mci1).reshape(N, 4)[inter]
    in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
    mco2, out2t = code.update_both(mci2, in2)
    out2 = out2t.reshape(k, Aq)[deinter].flatten()

    out = (out1 * out2 * mbi.flatten()).reshape(k, Aq)
#  print out1.reshape(k, 2),  out2.reshape(k, 2)
    mbo = (out.flatten()/ (out.sum(axis=1).repeat(Aq))).reshape(k, Aq)
    mco = np.zeros([p, Aq], float)
    mco[::2, :] = mco1.reshape(k, Aq)[map_inds_c1, :]
    mco[1::2, :] = mco2.reshape(k, Aq)[map_inds_c2, :]
    return np.vstack([mbo, mco])

def compare_performance(SNR):
  K = 30
  Aq = 2
  K1 = 3
  C = Turbo_Coder(7680, rate=1./2, inter=[], deinter=[], K1 = 3, Aq = 2, eff = 0, track = 2, K = 30); C.test(SNR);
  Ceff = Turbo_Coder(7680, rate=1./2, inter=np.uint32(C.inter), deinter=np.uint32(C.deinter), \
          K1 = 3, Aq = 2, eff = 1, track = 0, K = 30)
  
  mu32 = np.float32(C.mu)
  start_eff = time.time()
  mu_eff = decode_eff_test(mu32, K, C.k, C.p, 2, Ceff.inter, Ceff.deinter, Ceff.S, Ceff.Q, Ceff.map_inds_c1, Ceff.map_inds_c2)
  end_eff = time.time()
  print '\nEfficient implementation:'
  C.err_eval(C.mu, mu_eff,  C.b)
   
  start_ref = time.time()
  mu = decode_test(C.mu, C.K, C.k, C.p, 2, C.inter, C.deinter, Ceff.S, Ceff.Q, C.map_inds_c1,  C.map_inds_c2,C.code)
  end_ref = time.time()
  print '\nReference implementation:'
  C.err_eval(C.mu, mu, C.b)

  start_ref2 = time.time()
  C.decode(C.mu, 1e3)
  end_ref2 = time.time()
  print '\nReference 2 implementation:'
  C.err_eval(C.mu, mu, C.b)

  print 'reference time:', end_ref-start_ref, ',\teff time:', end_eff-start_eff,\
         'reference2 time:', end_ref2-start_ref2

def LLR_evaluator(S, Q):
  '''
  Evaluates LLR vector FSM update for given matrices S,Q
  '''
  Md, Ms = np.shape(S)
  Ns = int(np.round(np.log2(Ms)))
  Nd = int(np.round(np.log2(Md)))
  matr = np.zeros([Md * Ms, 2*Ns + 2*Nd], int)
  SS = [np.zeros([Md, Ms], int) for i in range(Ns)] # output matrix for s1, s2, ...
  for i in range(Ns):
    SS[i] = (S >> (Ns - 1 - i)) & 1
#  print SS
  for i in range(Md):
    for j in range(Ms):
      inds = dec2bin(j, Ns) # indices s1, s2, ...
      indd = dec2bin(i, Nd)
      indc = dec2bin(Q[i,j], Nd)
      matr[i * 2**Ns + j, :] = np.hstack([indd, inds, np.asarray([SS[k][i, j] for k in range(Ns)]),indc])
  return matr

def eval_LLR(matr):
  '''
    Prepare LLR update for a given matrix (binary data assumed)
    Example:
    S = np.zeros([2,8], int)
    Q = np.zeros([2,8], int)
    _Turbo_lib.create_matrix_SQ_MS(S,Q)
    matr = LLR_evaluator(S,Q)
    eval_LLR(matr)
  '''
  N, M = matr.shape
  res = [['float_max'+str(N/2)+'(' for i in range(2)] for j in range(M)]
  cnt = np.zeros([M, 2], int)
  s_names = ['ld[i]'] + ['lf'+str(i)+'[i]' for i in range(1,M/2)] + \
            ['lb'+str(i)+'[i+1]' for i in range(1,M/2)] +['lc[i]']
  for i in range(N):
    for j in range(M):
#      tmp = np.nonzero(np.hstack([matr[i,:j], matr[i,(j+1):]]))[0]      
      tmp = np.nonzero(matr[i,:])[0]      
      if j == 0:
        s_a = 'ld'
      elif j == M/2:
        s_a = 'lc'
      elif j < M/2:
        s_a = 'lf'+str(j)
      elif j > M/2:
        s_a = 'lb'+str(j-M/2)
      print i,j, matr[i,j]
      if cnt[j, matr[i,j]] == 0:
        st_str = res[j][matr[i,j]]
      else:
        st_str = res[j][matr[i,j]] + ', '
        
      if len(tmp) == 0:
        res[j][matr[i,j]] =  st_str + '0 '
      else:
        res[j][matr[i,j]] = st_str + ('+'.join([s_names[k] for k in tmp if k != j]))

      cnt[j, matr[i,j]] += 1
  for i in range(2):
    for j in range(M):
      res[j][i] = res[j][i] + ')'  
  for i in range(M):
    print 'num = ', res[i][1]
    print 'denum = ', res[i][0]
    print s_names[i], ' = num - denum\n'
  return res
    

#S1 = S >> 1
#S2 = S & 1
#for i in range(2):
#    for j in range(2):
#        for k in range(2):
#            a.append([i, j ,k, S1[i, j + 2*k], S2[i, j + 2*k], Q[i, j + 2*k]])

if (__name__ == '__main__'):
  1+1
#  compare_performance() 
