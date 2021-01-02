#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void update_FSM(int N, double *pc, double *pd, double *out) {
//  double ps1 [N+1];
//  double ps [N+1];
  double *ps1 = (double *) malloc(sizeof(double) * (N+1));
  double *ps = (double *) malloc(sizeof(double) * (N+1));
  ps[0] = 1;
  ps1[N] = 0.5;
  double p0, p1;
  int i;
  for (i =0 ; i < N; i++){ // Forward
    p0 = ps[i] * pc[i] * pd[i] + (1-ps[i])*pc[i]*(1-pd[i]);
    p1 = ps[i]*(1-pc[i])*(1-pd[i]) + (1-ps[i]) * (1-pc[i]) * (pd[i]);
    ps[i + 1] = p0 / (p0 + p1);
//    printf("%f ", ps[i+1]);
  }
  for (i = N - 1; i >= 0; i--) {// Backward
    p0 = ps1[i+1] * pc[i] * pd[i] + (1-ps1[i+1])*(1-pc[i])*(1-pd[i]);
    p1 = (1-ps1[i+1])*(1-pc[i])*(pd[i]) + ps1[i+1] * pc[i] * (1-pd[i]);
    ps1[i] = p0 / (p0 + p1);
  }
//  for (i =1 ; i < N+1; i++)  printf("%f ", ps1[i]);
//  printf("\n");
  for (i =0 ; i < N; i++){ // to data
    p0 = ps[i] * ps1[i+1] * pc[i] + (1-ps[i]) * (1-ps1[i+1]) * (1-pc[i]);
    p1 = ps[i] * (1-ps1[i+1]) * (1-pc[i]) + (1-ps[i]) * ps1[i+1] * pc[i];
    out[i] = p0 / (p0 + p1);
  }
  free(ps1);  free(ps);
}

// The same as update_FSM_GFq for q=4, however unwrapping loops increases performance
void update_FSM_GF4(int N, double *pcv, double *pdv, double *out) { 
//  double ps1v [4*N+4];
//  double psv [4*N+4];
  double *ps1v = (double *) malloc(sizeof(double) * (4*N+4));
  double *psv = (double *) malloc(sizeof(double) * (4*N+4));
  psv[0] = 1;  psv[1] = 0;  psv[2] = 0;  psv[3] = 0;
  ps1v[4*N] = 0.25;  ps1v[4*N+1] = 0.25;  ps1v[4*N+2] = 0.25;  ps1v[4*N+3] = 0.25;
  double *pd, *pc, *ps, *ps1;
  double sum;
  int i;//int j;
  for (i =0 ; i < N; i++){ // Forward
    pd = &(pdv[4*i]); pc = &(pcv[4*i]);    ps = &(psv[4*i]); 
    ps[4] = ps[0]*pc[0]*pd[0] + ps[1]*pc[0]*pd[1] + ps[2]*pc[0]*pd[2] + ps[3]*pc[0]*pd[3];
    ps[5] = ps[0]*pc[1]*pd[1] + ps[1]*pc[1]*pd[0] + ps[2]*pc[1]*pd[3] + ps[3]*pc[1]*pd[2];
    ps[6] = ps[0]*pc[2]*pd[2] + ps[1]*pc[2]*pd[3] + ps[2]*pc[2]*pd[0] + ps[3]*pc[2]*pd[1];
    ps[7] = ps[0]*pc[3]*pd[3] + ps[1]*pc[3]*pd[2] + ps[2]*pc[3]*pd[1] + ps[3]*pc[3]*pd[0];
    sum = ps[4] + ps[5] + ps[6] + ps[7];
//    if (sum == 0) {ps[4] = 0.25; ps[5] = 0.25; ps[6] = 0.25; ps[7] = 0.25;}
//    else 
    ps[4] = ps[4]/sum; ps[5] = ps[5]/sum; ps[6] = ps[6]/sum;  ps[7] = ps[7]/sum;
//    printf("%f ", ps[i+1]);
  }
//  for (i =0 ; i < N + 1; i++) {
//    for (j =0 ; j < 4; j++) printf("%f ", psv[4*i + j]);
//    printf("\n");
//  }
//  printf("\n");
  for (i = N - 1; i >= 0; i--) {// Backward
    pd = &(pdv[4*i]); pc = &(pcv[4*i]);    ps1 = &(ps1v[4*i]);
    ps1[0] = ps1[4]*pc[0]*pd[0] + ps1[5]*pc[1]*pd[1] + ps1[6]*pc[2]*pd[2] + ps1[7]*pc[3]*pd[3];
    ps1[1] = ps1[4]*pc[0]*pd[1] + ps1[5]*pc[1]*pd[0] + ps1[6]*pc[2]*pd[3] + ps1[7]*pc[3]*pd[2];
    ps1[2] = ps1[4]*pc[0]*pd[2] + ps1[5]*pc[1]*pd[3] + ps1[6]*pc[2]*pd[0] + ps1[7]*pc[3]*pd[1];
    ps1[3] = ps1[4]*pc[0]*pd[3] + ps1[5]*pc[1]*pd[2] + ps1[6]*pc[2]*pd[1] + ps1[7]*pc[3]*pd[0];
    sum = ps1[0]+ps1[1]+ps1[2]+ps1[3];
//    if (sum == 0) {ps1[0] = 0.25; ps1[1] = 0.25; ps[2] = 0.25; ps[3] = 0.25;}
//    else {ps1[0] = ps1[0]/sum; ps1[1] = ps1[1]/sum; ps1[2] = ps1[2]/sum;  ps1[3] = ps1[3]/sum;}
    ps1[0] = ps1[0]/sum; ps1[1] = ps1[1]/sum; ps1[2] = ps1[2]/sum;  ps1[3] = ps1[3]/sum;
  }
//  for (i =1 ; i < N+1; i++)  printf("%f ", ps1[i]);
//  printf("\n");
  for (i =0 ; i < N; i++){ // to data
    pd = &(out[4*i]); pc = &(pcv[4*i]);    ps = &(psv[4*i]);  ps1 = &(ps1v[4*i]);
    pd[0] = ps1[4]*pc[0]*ps[0] + ps1[5]*pc[1]*ps[1] + ps1[6]*pc[2]*ps[2] + ps1[7]*pc[3]*ps[3];
    pd[1] = ps1[5]*pc[1]*ps[0] + ps1[4]*pc[0]*ps[1] + ps1[7]*pc[3]*ps[2] + ps1[6]*pc[2]*ps[3];
    pd[2] = ps1[6]*pc[2]*ps[0] + ps1[7]*pc[3]*ps[1] + ps1[4]*pc[0]*ps[2] + ps1[5]*pc[1]*ps[3];
    pd[3] = ps1[7]*pc[3]*ps[0] + ps1[6]*pc[2]*ps[1] + ps1[5]*pc[1]*ps[2] + ps1[4]*pc[0]*ps[3];
    sum = pd[0] + pd[1] + pd[2] + pd[3];
//    if (sum == 0) {pd[0]=0.25; pd[1]=0.25; pd[2]=0.25; pd[3]=0.25;}
//    else {}
    pd[0] = pd[0]/sum; pd[1] = pd[1]/sum; pd[2] = pd[2]/sum;  pd[3] = pd[3]/sum;
      
  }
  free(ps1v);
  free(psv);
}

// Update of the simplest recursive FSM upon GF(2^k) = GF(q)
void update_FSM_GFq(int N, double *pcv, double *pdv, double *out, long q) {
  int i;int j;int k;
  double *ps1v = (double *) malloc(sizeof(double) * (q*N+q));
  double *psv = (double *) malloc(sizeof(double) * (q*N+q));
  psv[0] = 1; for (i = 1; i < q; i++) psv[i] = 0; // zero phase shift
  for (i = 0; i < q; i++) ps1v[q*N + i] = 1.0/q; //unknown final state
  double *pd, *pc, *ps, *ps1;
  double sum;
  for (i =0 ; i < N; i++){ // Forward
    pd = &(pdv[q*i]); pc = &(pcv[q*i]);    ps = &(psv[q*i]); // actual q-tupples
    for (j = 0; j < q; j++) ps[q + j] = 0;
    for (j = 0; j < q; j++) {
      for (k = 0; k < q; k++) ps[q + (j^k)] += ps[j]*pd[k]*pc[j^k];
    }
    sum = 0;
    for (j = 0; j < q; j++) sum += ps[q + j]; // Norming
    for (j = 0; j < q; j++) ps[q + j] = ps[q + j]/sum;
  }

  for (i = N - 1; i >= 0; i--) {// Backward
    pd = &(pdv[q*i]); pc = &(pcv[q*i]);    ps1 = &(ps1v[q*i]);
    for (j = 0; j < q; j++) ps1[j] = 0;
    for (j = 0; j < q; j++) {
      for (k = 0; k < q; k++) ps1[j] += ps1[q + (j^k)]*pc[j^k]*pd[k];
    }
    sum = 0;
    for (j = 0; j < q; j++) sum += ps1[j];
    for (j = 0; j < q; j++) ps1[j] = ps1[j]/sum;
  }

  for (i =0 ; i < N; i++){ // to data
    pd = &(out[q*i]); pc = &(pcv[q*i]);    ps = &(psv[q*i]);  ps1 = &(ps1v[q*i]);
    for (j = 0; j < q; j++) pd[j] = 0;
    for (j = 0; j < q; j++) {
      for (k = 0; k < q; k++) pd[k] += ps1[q + (j^k)]*pc[j^k]*ps[j];
    }
    sum = 0;
    for (j = 0; j < q; j++) sum += pd[j];
    for (j = 0; j < q; j++) pd[j] = pd[j]/sum;
  }
  free(ps1v);
  free(psv);
}

// Update general FSM described by S and Q matrices -- possible to apply also on GFq with (Md = q)
// FSM of length N described by s1 = S(d,s) and c = Q(d,s) with cardinality Mx for x
void update_general_FSM(int N, double *pcv, double *pdv, double *out, long Md, long Ms, long Mc, long *S, long *Q) {
  int i;int j;int k; int index;
  // check import
/*  printf("N=%d\t Md=%ld\t Mc=%ld\t Ms=%ld\n", N, Md, Mc, Ms);
  for (i = 0; i < N*Mc;i++) printf("pc[%d]=%f ", i, pcv[i]);printf("\n");
  for (i = 0; i < N*Md;i++) printf("pd[%d]=%f ", i, pdv[i]);printf("\n");
  for (i = 0; i < Md;i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", S[index]);
    }
    printf("\n");
  }
  for (i = 0; i < Md; i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", Q[index]);
    }
    printf("\n");
  }*/

  double *ps1v = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to backward state messages
  double *psv = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to forward state messages
  psv[0] = 1; for (i = 1; i < Ms; i++) psv[i] = 0; // zero phase shift
  for (i = 0; i < Ms; i++) ps1v[Ms*N + i] = 1.0/Ms; //unknown final state
  double *pd, *pc, *ps, *ps1;
  double sum;
//  printf("ps_size = %ld\n", Ms*N+Ms);
  for (i =0 ; i < N; i++){ // Forward
//    printf("i = %d\n", i);
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    ps = &(psv[Ms*i]); // actual messages
    for (j = 0; j < Ms; j++) ps[Ms + j] = 0; //state n+1
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        ps[S[index]+Ms] += ps[j]*pd[k]*pc[Q[index]];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps[Ms + j]; // Norming
    for (j = 0; j < Ms; j++) ps[Ms + j] = ps[Ms + j]/sum;
//    for (j = 0; j < Ms; j++) printf("%f, ", ps[Ms + j]);    
//    printf("\n");
  }

  for (i = N - 1; i >= 0; i--) {// Backward
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Ms; j++) ps1[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[k];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps1[j];
    for (j = 0; j < Ms; j++) ps1[j] = ps1[j]/sum;
  //  for (j = 0; j < Ms; j++) printf("%f, ", ps1[j]);    
  //  printf("\n");
  }

  for (i =0 ; i < N; i++){ // to data
    pd = &(out[Md*i]); pc = &(pcv[Mc*i]);    ps = &(psv[Ms*i]);  ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Md; j++) pd[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        pd[k] += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
      }
    }
    sum = 0;
    for (j = 0; j < Md; j++) sum += pd[j];
    for (j = 0; j < Md; j++) pd[j] = pd[j]/sum;
  }
  free(ps1v);
  free(psv);
}

// Update general FSM described by S and Q matrices -- possible to apply also on GFq with (Md = q)
// FSM of length N described by s1 = S(d,s) and c = Q(d,s) with cardinality Mx for x
// Update is towards both codeword and data (e.g. for serially concetaneted turbo codes)
void update_general_FSM_both(int N, double *pcv, double *pdv, double *outc, double *outd, long Md, long Ms, long Mc, long *S, long *Q) {
  int i;int j;int k; int index;
  // check import
/*  printf("N=%d\t Md=%ld\t Mc=%ld\t Ms=%ld\n", N, Md, Mc, Ms);
  for (i = 0; i < N*Mc;i++) printf("pc[%d]=%f ", i, pcv[i]);printf("\n");
  for (i = 0; i < N*Md;i++) printf("pd[%d]=%f ", i, pdv[i]);printf("\n");
  for (i = 0; i < Md;i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", S[index]);
    }
    printf("\n");
  }
  for (i = 0; i < Md; i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", Q[index]);
    }
    printf("\n");
  }*/

  double *ps1v = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to backward state messages
  double *psv = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to forward state messages
  psv[0] = 1; for (i = 1; i < Ms; i++) psv[i] = 0; // zero initial state
  for (i = 0; i < Ms; i++) ps1v[Ms*N + i] = 1.0/Ms; //unknown final state
  double *pd, *pc, *ps, *ps1;
  double sum;
//  printf("ps_size = %ld\n", Ms*N+Ms);
  for (i =0 ; i < N; i++){ // Forward
//    printf("i = %d\n", i);
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    ps = &(psv[Ms*i]); // actual messages
    for (j = 0; j < Ms; j++) ps[Ms + j] = 0; //state n+1
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        ps[S[index]+Ms] += ps[j]*pd[k]*pc[Q[index]];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps[Ms + j]; // Norming
    for (j = 0; j < Ms; j++) ps[Ms + j] = ps[Ms + j]/sum;
//    for (j = 0; j < Ms; j++) printf("%f ", ps[Ms + j]);    
//    printf(" (%f) \t", sum);
  }
//  printf("\n");

  for (i = N - 1; i >= 0; i--) {// Backward
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Ms; j++) ps1[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[k];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps1[j];
    for (j = 0; j < Ms; j++) ps1[j] = ps1[j]/sum;
//    for (j = 0; j < Ms; j++) printf("%f ", ps1[j]);    
//    printf("\t");
  }
//  printf("\n");

  for (i =0 ; i < N; i++){ // to data
    pd = &(outd[Md*i]); pc = &(pcv[Mc*i]);    ps = &(psv[Ms*i]);  ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Md; j++) pd[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        pd[k] += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
      }
    }
    sum = 0;
    for (j = 0; j < Md; j++) sum += pd[j];
    for (j = 0; j < Md; j++) pd[j] = pd[j]/sum;
//    for (j = 0; j < Md; j++) printf("%f ", pd[j]);    
//    printf("\t");
  }
//  printf("\n");

  for (i =0 ; i < N; i++){ // to codewords
    pd = &(pdv[Md*i]); pc = &(outc[Mc*i]);    ps = &(psv[Ms*i]);  ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Mc; j++) pc[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        pc[Q[index]] += ps1[Ms + S[index]] * pd[k] * ps[j];
      }
    }
    sum = 0;
    for (j = 0; j < Mc; j++) sum += pc[j];
    for (j = 0; j < Mc; j++) pc[j] = pc[j]/sum;
//    for (j = 0; j < Mc; j++) printf("%f ", pc[j]);    
//    printf(" (%f) \t", sum);
  }
//  printf("\n");
  free(ps1v);
  free(psv);
}

// Update general FSM described by S and Q matrices -- possible to apply also on GFq with (Md = q)
// FSM of length N described by s1 = S(d,s) and c = Q(d,s) with cardinality Mx for x
// TODO
/*void update_general_FSM_reduced(int N, float thr, double *pcv, double *pdv, double *out, long Md, long Ms, long Mc, long *S, long *Q) {
  int i;int j;int k; int index;
  // check import
  printf("N=%d\t Md=%ld\t Mc=%ld\t Ms=%ld\n", N, Md, Mc, Ms);
  for (i = 0; i < N*Mc;i++) printf("pc[%d]=%f ", i, pcv[i]);printf("\n");
  for (i = 0; i < N*Md;i++) printf("pd[%d]=%f ", i, pdv[i]);printf("\n");
  for (i = 0; i < Md;i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", S[index]);
    }
    printf("\n");
  }
  for (i = 0; i < Md; i++) {
    for (j = 0; j < Ms; j++) {
      index = i + Md * j;
      printf("%ld ", Q[index]);
    }
    printf("\n");
  }

  double *ps1v = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to backward state messages
  double *psv = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages realtes to forward state messages

  double *ps1v_args = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Argument of Messages 
  double *ps1v_vals = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages related to the argument 
  int *ps1v_Nvals = (int *) malloc(sizeof(int) * (N+1)); // Number of used messages

  double *psv_args = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Argument of Messages 
  double *psv_vals = (double *) malloc(sizeof(double) * (Ms*N+Ms)); // Messages related to the argument 
  int *psv_Nvals = (int *) malloc(sizeof(int) * (N+1)); // Number of used messages
  int *psv1_Nvals = (int *) malloc(sizeof(int) * (N+1)); // Number of used messages
 
  psv_args[0] = 0; psv_vals[0] = 1; psv_Nvals[0] = 1; // zero initial state
  for (i = 1; i < N+1; i++) psv_Nvals[0] = Ms;

  for (i = 0; i < Ms; i++) { // unknown final state
    psv1_args[i] = i; psv1_vals[i] = i/Ms; 
  }
  for (i = 0; i < N+1; i++) psv1_Nvals[0] = Ms;
  
  long sum_psv1 = 0; long sum_psv = 0; // Cumulative sum

  double *pd, *pc, *ps, *ps1;
  double sum; 
  int Ms_used; int Ms2_used;
//  printf("ps_size = %ld\n", Ms*N+Ms);
  for (i =0 ; i < N; i++){ // Forward
//    printf("i = %d\n", i);
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    
    ps_vals = &(psv_vals[sum_psv]);  ps_args = &(psv_args[sum_psv]); 
    Ms_used = ps1v_Nvals[i];  Ms2_used = ps1v_Nvals[i+1];
    for (j = 0; j < Ms2_used; j++) ps[Ms + j] = 0; //state n+1
    for (j = 0; j < Ms_used; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * ps_args[j];
        ps_vals[S[index]+Ms] += ps[j]*pd[k]*pc[Q[index]];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps[Ms + j]; // Norming
    for (j = 0; j < Ms; j++) ps[Ms + j] = ps[Ms + j]/sum;
    
  }

  for (i = N - 1; i >= 0; i--) {// Backward
    pd = &(pdv[Md*i]); pc = &(pcv[Mc*i]);    ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Ms; j++) ps1[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[k];
      }
    }
    sum = 0;
    for (j = 0; j < Ms; j++) sum += ps1[j];
    for (j = 0; j < Ms; j++) ps1[j] = ps1[j]/sum;
  }

  for (i =0 ; i < N; i++){ // to data
    pd = &(out[Md*i]); pc = &(pcv[Mc*i]);    ps = &(psv[Ms*i]);  ps1 = &(ps1v[Ms*i]);
    for (j = 0; j < Md; j++) pd[j] = 0;
    for (j = 0; j < Ms; j++) {
      for (k = 0; k < Md; k++) {
        index = k + Md * j;
        pd[k] += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
      }
    }
    sum = 0;
    for (j = 0; j < Md; j++) sum += pd[j];
    for (j = 0; j < Md; j++) pd[j] = pd[j]/sum;
  }
  free(ps1v);
  free(psv);
}*/

// Efficient implementation of turbo decoding

void norm(float *vec, unsigned int len) {
  float sum;
  sum = 0; unsigned register j;
  for (j = len; j--;) sum += vec[j]; // Norming
  if (sum == 0) { // Avoiding NaNs resulting from all zero pdf
    sum = 1.0/len;
    for (j = len; j--;) vec[j] = sum;
  }
  else {
    sum = 1.0/sum;
    for (j = len; j--;) vec[j] = vec[j] * sum;
  }
}

void print_message(float *vec, unsigned int len) {
  unsigned register int j;
  for (j = 0; j < len; j++) printf("%f ", vec[j]);
  printf("\n");
}

void eff_turbo_update(unsigned int N, float *b, float *c1, float *c2, unsigned int Md, unsigned int Ms, unsigned int Mc, unsigned int *S, unsigned int *Q, unsigned int *inter, unsigned int *deinter, unsigned int K) {
  unsigned register int i;
  unsigned register int j;
  unsigned register int k; 
  unsigned register int l; 
  unsigned register int index;
  unsigned register int ind;
  float *i1o2 = (float *) malloc(sizeof(float) * (Md*N)); // input to C1, out from C2
  float *i2o1 = (float *) malloc(sizeof(float) * (Md*N)); // input to C2, out from C1
  float *ps1v = (float *) malloc(sizeof(float) * (Ms*N+Ms)); // Messages realtes to backward state messages
  float *psv = (float *) malloc(sizeof(float) * (Ms*N+Ms)); // Messages realtes to forward state messages
  float norm_state = 1.0/Ms;
 
  float *pd, *pc, *ps, *ps1, *od, *a; 
  
  for (i = N*Md ; i--;) i1o2[i] = b[i]; // initializing the input
  for (i = Ms; i--;) ps1v[Ms*N + i] = norm_state; //unknown final state
  for (i = Ms; i--;) {psv[i] = 0;}   // zero initial state
  psv[0] = 1; // zero initial state  

//  printf("Controlling input: Md = %d, Ms = %d, Mc = %d, K = %d\n",Md, Ms, Mc, K);
//  printf("b\tc1\tc2\n");
//  for (i = 0; i < 6; i++) { 
//    printf("[%.4f,%.4f] \t [%.4f,%.4f] \t [%.4f,%.4f]\n", b[2*i], b[2*i + 1], c1[2*i], c1[2*i+1], c2[2*i], c2[2*i+1]);
//  }

  for (k = K; k--;) { // K iterations
  // First SISO  
    for (i = 0 ; i < N; i++) { // Forward
      pd = &(i1o2[Md*i]); pc = &(c1[Mc*i]); ps = &(psv[Ms*i]);// actual messages
//      printf("i=%d\n", i);
      for (j = Ms; j--;) ps[Ms + j] = 0; //state n+1
      for (j = Ms; j--;) {
        for (l = Md; l--;) {
          index = l + Md * j;
          ps[S[index]+Ms] += ps[j]*pd[l]*pc[Q[index]];
//          printf("%.4f(s1:%d)+=%.4f(s:%d)*%.4f(d:%d)*%.4f(c:%d)\n", ps[S[index]+Ms], S[index]+Ms,  ps[j], j, pd[l], l,  pc[Q[index]], Q[index]);
        }
      }
      norm(&ps[Ms], Ms);
//      printf("\n");
    }
//    printf("SISO 1 s_n %d-th iteration:\n", K-k);
//    for (i = 0; i < N; i++) { 
//      printf("[%.4f,%.4f]\n", psv[2*i], psv[2*i + 1]);
//    }
  
    for (i = N; i--;) { // Backward and data
      pd = &(i1o2[Md*i]); pc = &(c1[Mc*i]);  ps1 = &(ps1v[Ms*i]);
      od = &(i2o1[Md*i]); // output to the second SISO 
      a =  &(b[Md*i]); ps = &(psv[Ms*i]);
      for (j = Ms; j--;) ps1[j] = 0;
      for (l = Md; l--;) {
        od[l] = 0;
        for (j = Ms; j--;) {
          index = l + Md * j;
          ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[l];
          od[l]  += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
        }
        od[l] *= a[l]; // mixing the soft info with a priory
      }
      norm(ps1, Ms);
      norm(od, Md);
//      printf("Writing %d to %d \n", i,  inter[i]);
    }
//    printf("SISO 2 input %d-th iteration:\n", K-k);
//    for (i = 0; i < N; i++) { 
//      printf("[%.4f,%.4f]\n", i2o1[2*inter[i]], i2o1[2*inter[i] + 1]);
//    }

// Second SISO  
    for (i =0 ; i < N; i++) { // Forward
      ind = inter[i];
      pd = &(i2o1[Md*ind]); // interleved SISO1 input mixed with data a priory
      pc = &(c2[Mc*i]); ps = &(psv[Ms*i]); // actual messages
      for (j = Ms; j--;) ps[Ms + j] = 0; //state n+1
      for (j = Ms; j--;) {
        for (l = Md; l--;) {
          index = l + Md * j;
          ps[S[index]+Ms] += ps[j]*pd[l]*pc[Q[index]];
        }
      }
      norm(&ps[Ms], Ms);
    }
  
    for (i = N; i--;) { // Backward and data
      ind = inter[i];
      pd = &(i2o1[Md*ind]); pc = &(c2[Mc*i]);  ps1 = &(ps1v[Ms*i]); ps = &(psv[Ms*i]);
      od = &(i1o2[Md*ind]); // deinterleaved input to first decoder
      a =  &(b[Md*ind]);
      for (j = Ms; j--;) ps1[j] = 0;
      for (l = Md; l--;) {
        od[l] = 0;
        for (j = Ms; j--;) {
          index = l + Md * j;
          ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[l];
          od[l]  += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
        }
        od[l] *= a[l]; // mixing the soft info with a priory
      }
      norm(ps1, Ms);
      norm(od, Md);
    }
//    printf("SISO 1 input %d-th iteration:\n", K-k);
//    for (i = 0; i < N; i++) { 
//      printf("[%.4f,%.4f]\n", i1o2[2*i], i1o2[2*i + 1]);
//    }
  }
  // Final iteration 


  // First SISO  
  for (i =0 ; i < N; i++) { // Forward (seems ok)
    pd = &(i1o2[Md*i]); pc = &(c1[Mc*i]); ps = &(psv[Ms*i]); // actual messages
    for (j = Ms; j--;) ps[Ms + j] = 0; //state n+1
    for (j = Ms; j--;) {
      for (l = Md; l--;) {
        index = l + Md * j;
        ps[S[index]+Ms] += ps[j]*pd[l]*pc[Q[index]];
      }
    }
    norm(&ps[Ms], Ms);
  }
   
  for (i = N; i--;) { // Backward, data and codeword 1
    pd = &(i1o2[Md*i]); pc = &(c1[Mc*i]);  ps1 = &(ps1v[Ms*i]); ps = &(psv[Ms*i]);
    od = &(i2o1[Md*i]); 
    a =  &(b[Md*i]);
    for (j = Ms; j--;) ps1[j] = 0;
    for (l = Md; l--;) {
      od[l] = 0;
      for (j = Ms; j--;) {
        index = l + Md * j;
        ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[l];
        od[l]  += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
//        printf("%f[%d] += %f[%d]*%f[%d]*%f[%d]\n", od[l], l, ps1[Ms + S[index]], Ms + S[index],  pc[Q[index]], Q[index] , ps[j], j);
      }
      od[l] *= a[l]; // mixing the soft info with a priory
    }
    for (l = Mc; l--;) pc[l] = 0;
    for (l = Md; l--;) {
      for (j = Ms; j--;) {
        index = l + Md * j;
        pc[Q[index]] += ps1[Ms + S[index]] * pd[l] * ps[j];
//        printf("%f[%d] += %f[%d]*%f[%d]*%f[%d]\n", pc[Q[index]], Q[index], ps1[Ms + S[index]], Ms + S[index], pd[l], l, ps[j], j);
      }
    }
    norm(pc, Mc);
    norm(od, Md);
    norm(ps1, Ms);
  }
//  printf("SISO 2 input %d-th iteration:\n", K-k);
//  for (i = 0; i < N; i++) { 
//    printf("[%.4f,%.4f]\n", i2o1[2*i], i1o2[2*i + 1]);
//  }
//  print_message(i2o1, (Md) * N);

  // Second SISO  
  for (i = Ms; i--;) {psv[i] = 0;} 
  psv[0] = 1; // zero initial state  
  for (i = Ms; i--;) ps1v[Ms*N + i] = norm_state; //unknown final state

  for (i =0 ; i < N; i++) { // Forward
    ind = inter[i];
    pd = &(i2o1[Md*ind]); pc = &(c2[Mc*i]); ps = &(psv[Ms*i]); // actual messages
    for (j = Ms; j--;) ps[Ms + j] = 0; //state n+1
    for (j = Ms; j--;) {
      for (l = Md; l--;) {
        index = l + Md * j;
        ps[S[index]+Ms] += ps[j]*pd[l]*pc[Q[index]];
//        printf("%f[%d] += %f[%d]*%f[%d]*%f[%d]\n", ps[S[index]+Ms], S[index]+Ms, ps[j], j, pd[l], l, pc[Q[index]], Q[index]);
      }
    }
    norm(&ps[Ms], Ms);
  }
  
//  print_message(psv, (Ms) * (N+1));

  for (i = N; i--;) { // Backward, data and codeword 2
    ind = inter[i];
    pd = &(i2o1[Md*ind]); ps = &(psv[Ms*i]);
    pc = &(c2[Mc*i]);  ps1 = &(ps1v[Ms*i]);
    od = &(i1o2[Md*ind]); // 
    a = &(b[Md*ind]);
    for (j = Ms; j--;) ps1[j] = 0;
   
    for (l = Md; l--;) {
      od[l] = 0;
      for (j = Ms; j--;) {
        index = l + Md * j;
        ps1[j] += ps1[Ms + S[index]] * pc[Q[index]] * pd[l];
        od[l]  += ps1[Ms + S[index]] * pc[Q[index]] * ps[j];
      }
      od[l] *= pd[l]; // mixing the soft info with a priory
    }
    for (l = Mc; l--;) pc[l] = 0;
    for (l = Md; l--;) {
      for (j = Ms; j--;) {
        index = l + Md * j;
        pc[Q[index]] += ps1[Ms + S[index]] * pd[l] * ps[j];
//        printf("%f[%d] += %f[%d]*%f[%d]*%f[%d]\n", pc[Q[index]], Q[index], ps1[Ms + S[index]], Ms + S[index], pd[l], l, ps[j], j);
      }
    }
    norm(pc, Mc);
    norm(od,  Md);
    norm(ps1, Ms);
  }
  for (i = N * Md; i--;) b[i] = i1o2[i];

//  printf("SISO 1 input %d-th iteration:\n", K-k);
//  for (i = 0; i < N; i++) { 
//    printf("[%.4f,%.4f]\n", i1o2[2*i], i1o2[2*i + 1]);
//  }

  free(ps1v); ps1v = NULL;
  free(psv); psv= NULL;
  free(i1o2); i1o2 = NULL;
  free(i2o1); i2o1 = NULL;
}


int test_update_general_FSM() {
  double pc [] = {0.2,0.8,0.3,0.7};
  double pd [] = {0.1,0.9,0.2,0.8};
  double *out = (double *) malloc(sizeof(double) * 4);
  double *outc = (double *) malloc(sizeof(double) * 4);
  double *outd = (double *) malloc(sizeof(double) * 4);
  double *p1 = (double *) malloc(sizeof(double) * 4);
  double *p2 = (double *) malloc(sizeof(double) * 4);
  long S [] = {0,1,1,0};
  long Q [] = {0,1,1,0};
  int i;
  for (i = 0; i < 4; i++) {
    p1[i] = pc[i]; p2[i] = pd[i];
  }
  for (i = 0; i < 4; i++) out[i]=0;
  printf("S=[%ld %ld %ld %ld]\n", S[0], S[1],S[2], S[3]);
  printf("Q=[%ld %ld %ld %ld]\npc:", Q[0], Q[1],Q[2], Q[3]);
  for (i = 0; i < 4; i++) printf("%f ", p1[i]);printf("\npd:");
  for (i = 0; i < 4; i++) printf("%f ", p2[i]);printf("\n");
//  update_FSM_GF4(5, p1, p2, out);
  update_general_FSM(2, p1, p2, out, 2, 2, 2, S, Q);
  update_general_FSM_both(2, p1, p2, outc, outd, 2, 2, 2, S, Q);
  for (i = 0; i < 4; i++) printf("%f ", out[i]);  printf("\n");
  for (i = 0; i < 4; i++) printf("%f ", outc[i]);  printf("\n");
  for (i = 0; i < 4; i++) printf("%f ", outd[i]);  printf("\n");
  free(out); free(p1); free(p2);
  free(outd); free(outc);
  return 0;
}

int test_eff_turbo_update() {
  float pc1 [] = {0.0078, 0.9922, 0.0021, 0.9979};
  float pc2 [] = {0.9988, 0.0012, 0.0035, 0.9965};
  float pd  [] = {0.0175, 0.9825, 0.9742, 0.0258};
  unsigned int inter [] = {1, 0};
  unsigned int deinter [] = {1, 0};
  unsigned int S [] = {0,1,1,0};
  unsigned int Q [] = {0,1,1,0};
  unsigned int K = 0;

  printf("pd = [%f %f], [%f %f]\n", pd[0], pd[1],pd[2], pd[3]);
  printf("pc1 = [%f %f], [%f %f]\n", pc1[0], pc1[1],pc1[2], pc1[3]);
  printf("pc2 = [%f %f], [%f %f]\n\n", pc2[0], pc2[1],pc2[2], pc2[3]);
  eff_turbo_update(2, pd, pc1, pc2, 2, 2, 2, S, Q, inter, deinter, K);
  printf("pd = [%f %f], [%f %f]\n", pd[0], pd[1],pd[2], pd[3]);
  printf("pc1 = [%f %f], [%f %f]\n", pc1[0], pc1[1],pc1[2], pc1[3]);
  printf("pc2 = [%f %f], [%f %f]\n", pc2[0], pc2[1],pc2[2], pc2[3]);
  return 0;
}

int main() {
  test_eff_turbo_update();
  test_update_general_FSM();
  return 0;
}

