#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void sparse_turbo_decode(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta);

void sparse_turbo_decode_MS(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta);

void sparse_turbo_decode_MS_zero(double *pc1, double *pc2, double *pd, long *res, long *inter, long  *deinter, long K, long N, long Aq, long Nd,long Mc, long Ns, double theta);

void update_FSM_sparse_eff(long Aq, long N, double thresh, double *pdv, double *pcv, double *out);

void update_FSM_sparse_eff_test(long Aq, long N, double thresh, double *pdv, double *pcv);

typedef struct { // For representing a single piece of pmf consisting of index and corresponding pmf such as p(index) = value
  int *index; // corresponding index (or vecrot of indecis)
  float value; // probability
} Value;


typedef struct { // (Sparse) representation of message 
  int max_length; // maximum number of pmf elements
  int act_length; // number of used pmf elements
  Value *vector; // pmf vector
} Message;

typedef struct { // Auxiliary struct for FN update evaluation
  int index;
  float *value;
} list;


