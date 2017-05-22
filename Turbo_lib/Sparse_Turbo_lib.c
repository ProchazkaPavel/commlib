#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <Sparse_Turbo_lib.h>


void copy(Value *a, Value *b){
  b->index = a->index;
  b->value = a->value;
}

void copy_full(Value *a, Value *b){
  b->index[0] = a->index[0];
  b->value = a->value;
}

void merge(Value *llist, int ln, Value *rlist, int rn, Value *result)
{
  int i = 0;  int j = 0;
  // Giving the lists together with respect to the final sort
  while ((i < ln) && (j < rn))
  {
    if (llist[i].value > rlist[j].value)  {copy(&(llist[i]), &(result[i + j])); i++;}
    else { copy(&(rlist[j]), &(result[i + j])); j++; }
  }
  // Copy the rest of the list
  if (i == ln) {while ((j < rn)){copy(&(rlist[j]),&(result[i + j]));j++;}}
  else {while ((i < ln)){copy(&(llist[i]), &(result[i + j]));i++;}}
  // Copy result to the original memory
  for (i = 0; i < ln; i++) {copy(&(result[i]), &(llist[i]));}
  for (j = 0; j < rn; j++) {copy(&(result[j + ln]), &(rlist[j]));}
}

Value *mergesort(Value *list, int length, Value *temp)
{
  if (length <= 1) {return list;} //already sorted
  else
  {
    Value *lList;
    Value *rList;
    int k = (int)(length / 2);

    //Splitting into rlist and llist
    lList = list;
    rList = &(list[k]);

    // Sort the sublists
    lList = mergesort(lList, k, temp);
    rList = mergesort(rList, length - k, temp);

    // Merge the sorted list together to one sort list
    merge(lList, k, rList, length - k, temp);
    return lList;
  }
}

void index_merge(Value *llist, int ln, Value *rlist, int rn, Value *result)
{
  int i = 0;  int j = 0;
  // Giving the lists together with respect to the final sort
  while ((i < ln) && (j < rn))
  {
    if (llist[i].index[0] < rlist[j].index[0])  {copy_full(&(llist[i]), &(result[i + j])); i++;}
    else { copy_full(&(rlist[j]), &(result[i + j])); j++; }
  }
  // Copy the rest of the list
  if (i == ln) {while ((j < rn)){copy_full(&(rlist[j]),&(result[i + j]));j++;}}
  else {while ((i < ln)){copy_full(&(llist[i]), &(result[i + j]));i++;}}
  // Copy result to the original memory
  for (i = 0; i < ln; i++) {copy_full(&(result[i]), &(llist[i]));}
  for (j = 0; j < rn; j++) {copy_full(&(result[j + ln]), &(rlist[j]));}
}

Value *index_mergesort(Value *list, int length, Value *temp)
{
  if (length <= 1) {return list;} //already sorted
  else
  {
    Value *lList;
    Value *rList;
    int k = (int)(length / 2);

    //Splitting into rlist and llist
    lList = list;
    rList = &(list[k]);

    // Sort the sublists
    lList = index_mergesort(lList, k, temp);
    rList = index_mergesort(rList, length - k, temp);

    // Merge the sorted list together to one sort list
    index_merge(lList, k, rList, length - k, temp);
    return lList;
  }
}

void norm(Message *m) {
  int i; float sum = 0;
  for (i = 0; i < m->act_length; i++) sum += (&(m->vector[i]))->value;
  if (sum > 1e-6) {
    float mult = 1.0 / sum;
    for (i = 0; i < m->act_length; i++) (&(m->vector[i]))->value = (&(m->vector[i]))->value * mult;
  }
  else m->act_length = 0;
}

void sort(Message *m, Value *temp) { // Sort 
  m->vector = mergesort(m->vector, m->act_length, temp);
}

void select_active(Message *m, float threshold) {
  int i;
  for (i = 0; i < m->act_length; i++) {
    if ((&(m->vector[i]))->value < threshold) break;
  }
  m->act_length = i;
}

Message *alloc_message(int max_length, int num_messages, int dim) {
  Message *m = (Message *) malloc(sizeof(Message) * num_messages);
  Value *v;
  int i; int j;
  for (i = 0; i < num_messages; i++) {
    v = (Value *) malloc(sizeof(Value) * max_length);
    for (j = 0; j < max_length; j++) {
      (&(v[j]))->index = (int *) malloc(sizeof(int) * dim);
    }
    (&(m[i]))->vector = v;
    (&(m[i]))->max_length = max_length;
    (&(m[i]))->act_length = 0; // Void message by default
  }
  return m;
}

void clear_message(Message *m, int num_messages) {
  int i; int j;
  for (i = 0; i < num_messages; i++) {
    for (j = 0; j < (&(m[i]))->max_length; j++) {
      free((&((&(m[i]))->vector[j]))->index);
    }
    free((&(m[i]))->vector);
    (&(m[i]))->vector = NULL;
  }
  free(m); m = NULL; 
}

void print_message(Message *m, int num_messages, int dim) {
  printf("Printing %d messages:\n", num_messages);
  int i, j, k;
  Value *v;
  for (i = 0; i < num_messages; i++) {
    printf("%d-th message of length %d:\n", i, (&(m[i]))->act_length);
    for (j =0; j < (&(m[i]))->act_length ; j++){
      v = (&((&(m[i]))->vector[j]));
      for (k = 0; k < dim; k++) printf("%d, ", (v->index)[k]);
      printf("%f, %p\n", v->value, v);
    }
  }
  printf("\n\n");
}

list *alloc_list(int len) {  
  list *l = (list *) malloc(sizeof(list) * len);
  return l;
}

int process_update(int ii, int index, float val, int *iv, int dim, list *l, float thr_act, Message *m, int N) {
  int j = 0; 
  int i;
  for (j = 0; j < ii; j++) { // Test if the update index already appears
    if (index == (&(l[j]))->index) {
      *((&(l[j]))->value) += val; // if so, append the corresponding probability    
//                printf("Test Passed, index already in list:%ld, %f\n", index, vs1);
      break;
    }
  }
//  if ((j == ii) && (val > thr_act)) { // new index (corresponding probability must be large enough)
  if ((j == ii) && (val > 1e-8)) { // new index (corresponding probability must be large enough)
    if (ii == N) { // full stack of indecis (must overwrite one of the current)
      j = 0;
      while ((j < N) && (val < *((&(l[j]))->value))) j++; // find the index, where the value is smaller than the actual one
      if (j < N) {
        (&(l[j]))->value = &((&(m->vector[j]))->value); // adjust the poiter to proper value
        (&(l[j]))->index = index;
        (&(m->vector[j]))->value = val; // set the update
        for (i = 0; i < dim; i++) (&(m->vector[j]))->index[i] = iv[i];
      }
    }
    else {
      (&(l[j]))->index = index; // add it to the list
      (&(l[j]))->value = &((&(m->vector[ii]))->value); // adjust the poiter to proper value
      (&(m->vector[ii]))->value = val; // set the update
      for (i = 0; i < dim; i++) (&(m->vector[ii]))->index[i] = iv[i]; // set the update
      ii++; // increase the active size
    }
  }
  return ii;
}

void update_FSM_sparse_eff(long Aq, long N, double thresh, double *pdv, double *pcv, double *out) {
  long Ms = Aq*Aq*Aq;
  long Aq2 = Aq * Aq;
  double *ps1v = (double *) malloc(sizeof(double) * (Ms*(N+1))); // Messages realtes to backward state messages
  double *psv = (double *) malloc(sizeof(double) * (Ms*(N+1))); // Messages realtes to forward state messages
  double *psf0 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psf1 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psf2 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb0 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb1 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb2 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  //out = (double *) malloc(sizeof(double) * (Aq*N)); // Messages realtes to forward state messages
  int i,j, s0i, s1i, s2i, di, ci, si, Si, S0i, S1i, S2i;
//  for (i = 0; i < N*Aq;i++) printf("pc[%d]=%f ", i, pcv[i]);printf("\n");
//  for (i = 0; i < N*Aq;i++) printf("pd[%d]=%f ", i, pdv[i]);printf("\n");
//  printf("N:%ld, Aq:%ld, Aq2:%ld, Ms:%ld, th:%lf\n", N, Aq, Aq2, Ms, thresh);

  psv[0] = 1; for (i = 1; i < (N+1)*Ms; i++) psv[i] = 0; // zero init state
  psf0[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf0[i] = 0; // zero init state
  psf1[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf1[i] = 0; // zero init state
  psf2[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf2[i] = 0; // zero init state
  for (i = 0; i <(N+1)* Aq; i++) {psb2[i] = 1.0/Aq;psb1[i] = 1.0/Aq;psb0[i] = 1.0/Aq;} // unknow fin. state
  for (i = 0; i < Ms; i++) ps1v[Ms*N + i] = 1.0/Ms; //unknown final state
  double *pd, *pc, *ps, *ps1, *sf1, *sf0, *sf2, *sb1, *sb0, *sb2;
  double sum; double val;
  int *i0 = (int *) malloc(sizeof(int) * Aq);int cnt0=0;
  int *i1 = (int *) malloc(sizeof(int) * Aq);int cnt1=0;
  int *i2 = (int *) malloc(sizeof(int) * Aq);int cnt2=0;
  int *ic = (int *) malloc(sizeof(int) * Aq);int cntc=0;
  int *id = (int *) malloc(sizeof(int) * Aq);int cntd=0;
  // Forward
  for (i = 0; i < N; i++) {
   // printf("i=%d\n\n",i);
    pd = &(pdv[Aq*i]);   pc = &(pcv[Aq*i]);    ps = &(psv[Ms*i]); // actual messages
    sf0 = &(psf0[Aq*i]); sf1 = &(psf1[Aq*i]);  sf2 = &(psf2[Aq*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sf0[j] > thresh) i0[cnt0++] = j;
      if (sf1[j] > thresh) i1[cnt1++] = j;
      if (sf2[j] > thresh) i2[cnt2++] = j;
      if (pd[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
 //   printf("starting Forward (%d,%d,%d,%d)\n", cnt0, cnt1, cnt2, cntd);
    for (s0i = 0; s0i < cnt0; s0i++) {
      for (s1i = 0; s1i < cnt1; s1i++) {
        S2i = i1[s1i];
        for (s2i = 0; s2i < cnt2; s2i++) {
          S1i = (i0[s0i]+ i2[s2i]) % Aq;
          for (di = 0; di < cntd; di++) {
            ci = (i0[s0i] + i2[s2i]+ id[di]) % Aq;
            S0i = (id[di] + i2[s2i]) % Aq;
            Si = S2i*Aq2 + Aq * S1i + S0i;
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps[si] * pd[di] * pc[ci];
            ps[Ms + Si] += val;
            sum += val;
            sf2[Aq + S2i] += val;
            sf1[Aq + S1i] += val;
            sf0[Aq + S0i] += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Ms; j++) ps[j+Ms] = ps[j+Ms]/sum;
    for (j = 0; j < Aq; j++) sf0[j+Aq] = sf0[j+Aq]/sum;
    
    for (j = 0; j < Aq; j++) sf1[j+Aq] = sf1[j+Aq]/sum;
    for (j = 0; j < Aq; j++) sf2[j+Aq] = sf2[j+Aq]/sum;
 //   for (j = 0; j < Ms; j++) printf("%f, ", ps[j+Ms]);
 //   printf("\n");
  }
  // Backward
  for (i = N-1 ; i >= 0; i--){ 
   // printf("i=%d\n\n",i);
    pd = &(pdv[Aq*i]);   pc = &(pcv[Aq*i]);    ps1 = &(ps1v[Ms*i]); // actual messages
    sb0 = &(psb0[Aq*i]); sb1 = &(psb1[Aq*i]);  sb2 = &(psb2[Aq*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sb0[j+Aq] > thresh) i0[cnt0++] = j;
      if (sb1[j+Aq] > thresh) i1[cnt1++] = j;
      if (sb2[j+Aq] > thresh) i2[cnt2++] = j;
      if (pd[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
 //   printf("starting Bakward (%d,%d,%d,%d)\n", cnt0, cnt1, cnt2, cntd);
    for (S2i = 0; S2i < cnt2; S2i++) {
      s1i = i2[S2i];
      for (S0i = 0; S0i < cnt0; S0i++) {
        for (S1i = 0; S1i < cnt1; S1i++) {
          for (di = 0; di < cntd; di++) {
            s2i = (Aq + i0[S0i]- id[di]) % Aq;
            s0i = (Aq + i1[S1i]- i0[S0i] + id[di]) % Aq;
            ci = (i1[S1i] + id[di]) % Aq;
            Si = S2i*Aq2 + Aq * S1i + S0i;
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps1[Si + Ms] * pd[di] * pc[ci];
            ps1[si] += val;
            sum += val;
            sb2[S2i] += val;
            sb1[S1i] += val;
            sb0[S0i] += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Ms; j++) ps1[j] = ps1[j]/sum;
    for (j = 0; j < Aq; j++) sb0[j] = sb0[j]/sum;    
    for (j = 0; j < Aq; j++) sb1[j] = sb1[j]/sum;
    for (j = 0; j < Aq; j++) sb2[j] = sb2[j]/sum;
 //   for (j = 0; j < Ms; j++) printf("%f, ", ps1[j]);
 //   printf("\n");
  }

  // To data
  
  for (i = 0; i < N; i++) {
  
    
   // printf("i=%d\n\n",i);
    pd = &(out[Aq*i]);   pc = &(pcv[Aq*i]);    ps = &(psv[Ms*i]); // actual messages
    sf0 = &(psf0[Aq*i]); sf1 = &(psf1[Aq*i]);  sf2 = &(psf2[Aq*i]); // actual messages
    sb0 = &(psb0[Aq + Aq*i]); ps1 = &(ps1v[Ms + Ms*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sf0[j] > thresh) i0[cnt0++] = j;
      if (sf1[j] > thresh) i1[cnt1++] = j;
      if (sf2[j] > thresh) i2[cnt2++] = j;
      if (sb0[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
  
    for (s0i = 0; s0i < cnt0; s0i++) {
      for (s1i = 0; s1i < cnt1; s1i++) {
        S2i = i1[s1i];
        for (s2i = 0; s2i < cnt2; s2i++) {
          S1i = (i0[s0i]+ i2[s2i]) % Aq;
          for (S0i = 0; S0i < cntd; S0i++) {
            di = (Aq + id[S0i] - i2[s2i]) % Aq;
            ci = (i0[s0i] + i2[s2i]+ di) % Aq;
            Si = S2i*Aq2 + Aq * S1i + id[S0i];
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps[si] * ps1[Si] * pc[ci];
            pd[di] += val;
            sum += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Aq; j++) pd[j] = pd[j]/sum;
  }
  free(i0);
  free(i1);
  free(i2);
  free(ic);
  free(id);
  free(ps1v);
  free(psv);
  free(psf0);
  free(psf1);
  free(psf2);
}
/*
                      Si = S2i*Aq**2 + Aq * S1i + S0i
                      si = s2i*Aq**2 + Aq * s1i + s0i
                      val = sf[i, si] * pd[i, di] * pc[i, ci]
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
    
*/

void update_FSM_sparse_eff_test(long Aq, long N, double thresh, double *pdv, double *pcv) {
  long Ms = Aq*Aq*Aq;
  long Aq2 = Aq * Aq;
  double *ps1v = (double *) malloc(sizeof(double) * (Ms*(N+1))); // Messages realtes to backward state messages
  double *psv = (double *) malloc(sizeof(double) * (Ms*(N+1))); // Messages realtes to forward state messages
  double *psf0 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psf1 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psf2 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb0 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb1 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *psb2 = (double *) malloc(sizeof(double) * (Aq*(N+1))); // Messages realtes to forward state messages
  double *out = (double *) malloc(sizeof(double) * (Aq*N)); // Messages realtes to forward state messages
  int i,j, s0i, s1i, s2i, di, ci, si, Si, S0i, S1i, S2i;
//  for (i = 0; i < N*Aq;i++) printf("pc[%d]=%f ", i, pcv[i]);printf("\n");
//  for (i = 0; i < N*Aq;i++) printf("pd[%d]=%f ", i, pdv[i]);printf("\n");
//  printf("N:%ld, Aq:%ld, Aq2:%ld, Ms:%ld, th:%lf\n", N, Aq, Aq2, Ms, thresh);

  psv[0] = 1; for (i = 1; i < (N+1)*Ms; i++) psv[i] = 0; // zero init state
  psf0[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf0[i] = 0; // zero init state
  psf1[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf1[i] = 0; // zero init state
  psf2[0] = 1; for (i = 1; i <(N+1)* Aq; i++) psf2[i] = 0; // zero init state
  for (i = 0; i <(N+1)* Aq; i++) {psb2[i] = 1.0/Aq;psb1[i] = 1.0/Aq;psb0[i] = 1.0/Aq;} // unknow fin. state
  for (i = 0; i < Ms; i++) ps1v[Ms*N + i] = 1.0/Ms; //unknown final state
  double *pd, *pc, *ps, *ps1, *sf1, *sf0, *sf2, *sb1, *sb0, *sb2;
  double sum; double val;
  int *i0 = (int *) malloc(sizeof(int) * Aq);int cnt0=0;
  int *i1 = (int *) malloc(sizeof(int) * Aq);int cnt1=0;
  int *i2 = (int *) malloc(sizeof(int) * Aq);int cnt2=0;
  int *ic = (int *) malloc(sizeof(int) * Aq);int cntc=0;
  int *id = (int *) malloc(sizeof(int) * Aq);int cntd=0;
  // Forward
  for (i = 0; i < N; i++) {
   // printf("i=%d\n\n",i);
    pd = &(pdv[Aq*i]);   pc = &(pcv[Aq*i]);    ps = &(psv[Ms*i]); // actual messages
    sf0 = &(psf0[Aq*i]); sf1 = &(psf1[Aq*i]);  sf2 = &(psf2[Aq*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sf0[j] > thresh) i0[cnt0++] = j;
      if (sf1[j] > thresh) i1[cnt1++] = j;
      if (sf2[j] > thresh) i2[cnt2++] = j;
      if (pd[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
 //   printf("starting Forward (%d,%d,%d,%d)\n", cnt0, cnt1, cnt2, cntd);
    for (s0i = 0; s0i < cnt0; s0i++) {
      for (s1i = 0; s1i < cnt1; s1i++) {
        S2i = i1[s1i];
        for (s2i = 0; s2i < cnt2; s2i++) {
          S1i = (i0[s0i]+ i2[s2i]) % Aq;
          for (di = 0; di < cntd; di++) {
            ci = (i0[s0i] + i2[s2i]+ id[di]) % Aq;
            S0i = (id[di] + i2[s2i]) % Aq;
            Si = S2i*Aq2 + Aq * S1i + S0i;
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps[si] * pd[di] * pc[ci];
            ps[Ms + Si] += val;
            sum += val;
            sf2[Aq + S2i] += val;
            sf1[Aq + S1i] += val;
            sf0[Aq + S0i] += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Ms; j++) ps[j+Ms] = ps[j+Ms]/sum;
    for (j = 0; j < Aq; j++) sf0[j+Aq] = sf0[j+Aq]/sum;
    
    for (j = 0; j < Aq; j++) sf1[j+Aq] = sf1[j+Aq]/sum;
    for (j = 0; j < Aq; j++) sf2[j+Aq] = sf2[j+Aq]/sum;
 //   for (j = 0; j < Ms; j++) printf("%f, ", ps[j+Ms]);
 //   printf("\n");
  }
  // Backward
  for (i = N-1 ; i >= 0; i--){ 
   // printf("i=%d\n\n",i);
    pd = &(pdv[Aq*i]);   pc = &(pcv[Aq*i]);    ps1 = &(ps1v[Ms*i]); // actual messages
    sb0 = &(psb0[Aq*i]); sb1 = &(psb1[Aq*i]);  sb2 = &(psb2[Aq*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sb0[j+Aq] > thresh) i0[cnt0++] = j;
      if (sb1[j+Aq] > thresh) i1[cnt1++] = j;
      if (sb2[j+Aq] > thresh) i2[cnt2++] = j;
      if (pd[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
 //   printf("starting Bakward (%d,%d,%d,%d)\n", cnt0, cnt1, cnt2, cntd);
    for (S2i = 0; S2i < cnt2; S2i++) {
      s1i = i2[S2i];
      for (S0i = 0; S0i < cnt0; S0i++) {
        for (S1i = 0; S1i < cnt1; S1i++) {
          for (di = 0; di < cntd; di++) {
            s2i = (Aq + i0[S0i]- id[di]) % Aq;
            s0i = (Aq + i1[S1i]- i0[S0i] + id[di]) % Aq;
            ci = (i1[S1i] + id[di]) % Aq;
            Si = S2i*Aq2 + Aq * S1i + S0i;
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps1[Si + Ms] * pd[di] * pc[ci];
            ps1[si] += val;
            sum += val;
            sb2[S2i] += val;
            sb1[S1i] += val;
            sb0[S0i] += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Ms; j++) ps1[j] = ps1[j]/sum;
    for (j = 0; j < Aq; j++) sb0[j] = sb0[j]/sum;    
    for (j = 0; j < Aq; j++) sb1[j] = sb1[j]/sum;
    for (j = 0; j < Aq; j++) sb2[j] = sb2[j]/sum;
 //   for (j = 0; j < Ms; j++) printf("%f, ", ps1[j]);
 //   printf("\n");
  }

  // To data
  
  for (i = 0; i < N; i++) {
  
    
   // printf("i=%d\n\n",i);
    pd = &(out[Aq*i]);   pc = &(pcv[Aq*i]);    ps = &(psv[Ms*i]); // actual messages
    sf0 = &(psf0[Aq*i]); sf1 = &(psf1[Aq*i]);  sf2 = &(psf2[Aq*i]); // actual messages
    sb0 = &(psb0[Aq + Aq*i]); ps1 = &(ps1v[Ms + Ms*i]); // actual messages
    for (j = 0; j < Aq; j++) {
   //   printf("j:%d\n", j);
      if (sf0[j] > thresh) i0[cnt0++] = j;
      if (sf1[j] > thresh) i1[cnt1++] = j;
      if (sf2[j] > thresh) i2[cnt2++] = j;
      if (sb0[j] > thresh) id[cntd++] = j;
   //   printf("-----\n");
    }
    sum = 0;
  
    for (s0i = 0; s0i < cnt0; s0i++) {
      for (s1i = 0; s1i < cnt1; s1i++) {
        S2i = i1[s1i];
        for (s2i = 0; s2i < cnt2; s2i++) {
          S1i = (i0[s0i]+ i2[s2i]) % Aq;
          for (S0i = 0; S0i < cntd; S0i++) {
            di = (Aq + id[S0i] - i2[s2i]) % Aq;
            ci = (i0[s0i] + i2[s2i]+ di) % Aq;
            Si = S2i*Aq2 + Aq * S1i + id[S0i];
            si = i2[s2i]*Aq2 + Aq * i1[s1i] + i0[s0i];
     //       printf("Si:%d, si:%d\n", Si, si);
            val = ps[si] * ps1[Si] * pc[ci];
            pd[di] += val;
            sum += val;
  //          printf("pc:%lf, pd:%lf, ps:%lf\n", pc[i, ci],  pd[i, di], ps[i, si]);
    //        printf("Si:%d,si:%d,i0[s0i]:%d,i1[s1i]:%d,i2[s2i]:%d,S0i:%d, S1i:%d, S2i:%d,ic:%d, id:%d, val:%lf\n",Si,si,i0[s0i],i1[s1i],i2[s2i],S0i, S1i, S2i,ci, id[di], val);
          }
        }
      }
    }
    cnt0 = 0;
    cnt1 = 0;
    cnt2 = 0;
    cntd = 0;
    for (j = 0; j < Aq; j++) pd[j] = pd[j]/sum;
  }
  free(i0);
  free(i1);
  free(i2);
  free(ic);
  free(id);
  free(ps1v);
  free(psv);
  free(psf0);
  free(psf1);
  free(psf2);
  free(out);
}

void update_FSM_sparse(int N, int Ns, int Nd, int Nc, int Aq,  Message *pc, Message *pd, Message *out, Message *out1, float thr_act, int zero_term) {
  /*
   * Description:
   * ------------
   * Method to provide the FSM update for sparsely represented messages. The code is described by
   * the following set of equations (on bitwise-XOR)
   *
   * Forward (to sn+1):
   * 
   * all available:   |  no data       |  no codewords  |  no d no c      |  no sn
   * S0 = s2 + d      |  S0 = c - s0   |  S0 = s2 + d   |  S0 = all       |  S0, S2 = all 
   * S1 = s0 + s2     |  S1 = s0 + s2  |  S1 = s0 + s2  |  S1 = s0 + s2   |  S1 = c - d 
   * S2 = s1          |  S2 = s1       |  S2 = s1       |  S2 = s1        |  
   * c  = s0 + s2 + d |                |                |                 |
   *
   * Backward (to sn):
   * 
   * all available:   |  no data          |  no codewords     |  no d no c      |  no sn+1
   * s0 = S1 - S0 + d |  s0 = c - S0      |  s0 = S1 - S0 +d  |  s0 = S1 + s2   |  s0, s1 = all 
   * s1 = S2          |  s1 = S2          |  s1 = S2          |  s1 = S2        |  s2 = c - d -s0
   * s2 = S0 - d      |  s2 = S0 + S1 -c  |  s2 = S0 - d      |  s2 = all       |  
   * c  = S1 + d      |                   |                   |                 |
   *
   * To data (to d):
   * 
   * all available:   |  no codeword      |  no sn+1          |  no sn
   * d = S0 - s2      |  d = S0 - s2      |  d = c - s0 - s2  |  d = c - S1 
   * d = c - s0 - s2  |  S2 = s1          |                   |  
   * S2 = s1          |  S1 = s0 + s2     |                   |  
   * S1 = s0 + s2     |                   |                   |  
   *
   * To codewords (to c):
   * 
   * all available:   |  no data          |  no sn+1          |  no sn
   * c = d + s0 + s2  |  c = S0 + s0      |  c = d + s0 + s2  |  c = d + S1 
   * S0 = d + s2      |  S2 = s1          |                   |  
   * S2 = s1          |  S1 = s0 + s2     |                   |  
   * S1 = s0 + s2     |                   |                   |  
   *
   * Parameters:
   * -----------
   * N : int
   *   Length of data sequence (including the tail bits to achieve zero final state)
   *
   * Nd : int  
   *   Maximum possible considered data vector length
   *
   * Nc : int  
   *   Maximum possible considered codeword vector length
   *
   * Ns : int  
   *   Maximum possible considered state vector length
   *
   * Aq : int  
   *   Cardinality of data vector (not necessary the same as Nd) 
   *
   * pc : *Message
   *   Sparse message vector representing the codewords
   *
   * pd : *Message
   *   Sparse message vector representing the data
   *
   * out : *Message
   *   Sparse message vector representing for the data output
   *
   * out2 : *Message
   *   Sparse message vector representing for the codeword output (if NULL -> no codeword update is
   *   done)
   *
   * thr_act : float
   *   Probability threshold for considering the message active/inactive 
   *
   * zero_term : int
   *   Flag if the sequence terminates in the zero state (1 -> yes) (otherwise -> no)
   */

  int dim = 3; // Fixed for concrete FSM with constraint length K = 3
  int i, ii, di, ci, si, si1,  j;
  int id, ic;
  long index;
  long Aq2 = Aq*Aq;
  int ic_test;
  int is[3];
  int is1[3];
  float vd, vc, vs, vs1;
  Message *ps1 = alloc_message(Ns, N+1, dim); 
  Message *ps = alloc_message(Ns, N+1, dim); 
  list *l = alloc_list(Ns);
  // Known zero initial  state
  (&((&(ps[0]))->vector[0]))->value = 1;
  for (i = 0; i < dim; i++) (&((&(ps[0]))->vector[0]))->index[i] = 0;
  (&(ps[0]))->act_length = 1;

  if (zero_term == 1) {
    (&((&(ps1[N]))->vector[0]))->value = 1;
    for (i = 0; i < dim; i++) (&((&(ps1[N]))->vector[0]))->index[i] = 0;
    (&(ps1[N]))->act_length = 1;
  }
  else (&(ps1[N]))->act_length = 0;

  Value *temp = (Value *) malloc(sizeof(Value) * Ns);

  for (i = 0 ; i < N; i++){ // Forward
    ii = 0;
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
          vs = (&((&(ps[i]))->vector[si]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = is[2] ^ id; is1[1] = is[0] ^ is[2]; is1[2] = is[1]; // Next state
          ic_test = is[0] ^ is[2] ^ id;                                // Corresponding codeword
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
            ic = (&((&(pc[i]))->vector[ci]))->index[0];
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
            if (ic == ic_test){ // Find all compatible codewords
              vc = (&((&(pc[i]))->vector[ci]))->value;
              vs1 = vd * vc * vs; // update
              ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);
            }
          }
        }
      }
    }
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pd,ps available, pc unavailable
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
          vs = (&((&(ps[i]))->vector[si]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = is[2] ^ id; is1[1] = is[0] ^ is[2]; is1[2] = is[1]; // Next state
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          vs1 = vd * vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length == 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // pc, ps avail, pd unavail
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
        vs = (&((&(ps[i]))->vector[si]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
//          printf("IS:%d %d %d, i:%d, si:%d, ci:%d \n", is[0], is[1],is[2], i, si, ci); 
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = is[0] ^ ic; is1[1] = is[0] ^ is[2]; is1[2] = is[1]; // Next state
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
          vc = (&((&(pc[i]))->vector[ci]))->value;
          vs1 = vc * vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length == 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // only state available
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
        vs = (&((&(ps[i]))->vector[si]))->value;
        /*  ----------------  FSM description  ----------------------*/
        is1[1] = is[0] ^ is[2]; is1[2] = is[1]; // Next state
        /*  ---------------------------------------------------------*/
        for (si1 = 0; si1 < Aq; si1++) {
          is1[0] = si1; // All possibilities of the state
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          vs1 = vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // no state available
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          vc = (&((&(pc[i]))->vector[ci]))->value;
          vs1 = vc *vd; // update
          for (is1[0] = 0; is1[0] < Aq; is1[0]++) {
            for (is1[2] = 0; is1[2] < Aq; is1[2]++) {
              /*  ----------------  FSM description  ----------------------*/
              is1[1] = ic ^ id;
              /*  ---------------------------------------------------------*/
              index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
              ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
            }
          }
        }
      }
    }
  //    printf("Adjusting active length to %d\n", ii);
    (&(ps[i+1]))->act_length = ii;
    sort(&(ps[i+1]), temp);
    norm(&(ps[i+1]));  
    select_active(&(ps[i+1]), thr_act);        
  }
//  print_message(ps, N+1, 3);

  for (i = N ; i > 0; i--){ // Backward
    ii = 0;
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length > 0 )) { // all messages are available
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
//          printf("is1:%d %d %d\n", is1[0], is1[1], is1[2]);
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = is1[0] ^ is1[1] ^ id; is[1] = is1[2]; is[2] = is1[0] ^ id; // Previous state
          ic_test = is1[1] ^ id;                                      // Corresponding codeword
          /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
            ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
            if (ic == ic_test){ // Find all compatible codewords
              vc = (&((&(pc[i-1]))->vector[ci]))->value;
              vs = vd * vc * vs1; // update
              ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
            }
          }
        }
      }
    }
    if (((&(pd[i-1]))->act_length == 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length > 0 )) { // data are not available
      for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
        ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
        vc = (&((&(pc[i-1]))->vector[ci]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = is1[0] ^ ic; is[1] = is1[2]; is[2] = is1[0] ^ ic ^ is1[1]; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vc * vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length == 0 ) && ((&(ps1[i]))->act_length > 0 )) { // codewords are not available
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = is1[0] ^ is1[1] ^ id; is[1] = is1[2]; is[2] = is1[0] ^ id; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vd * vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length == 0 ) && ((&(pc[i-1]))->act_length == 0 ) && ((&(ps1[i]))->act_length > 0 )) { // neither data nor codewords are available
      for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];          
        vs1 = (&((&(ps1[i]))->vector[si1]))->value;
        is[1] = is1[2];
        for (is[2] = 0; is[2] < Aq; is[2]++) {
          /*  ----------------  FSM description  ----------------------*/
          is[0] = is1[1] ^ is[2]; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length == 0 )) { // sn+1 not available
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
          ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
          vc = (&((&(pc[i-1]))->vector[ci]))->value;
          for (is[1] = 0; is[1] < Aq; is[1]++) {
            for (is[2] = 0; is[2] < Aq; is[2]++) {
              /*  ----------------  FSM description  ----------------------*/
              is[0] = id ^ ic ^ is[2]; // Previous state
              /*  ---------------------------------------------------------*/
              index = is[0] + is[1] * Aq + is[2] * Aq2;
              vs = vd * vc; // update
              ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
            }
          }
        }
      }
    }
    (&(ps1[i-1]))->act_length = ii;
    sort(&(ps1[i-1]), temp);
    norm(&(ps1[i-1]));  
    select_active(&(ps1[i-1]), thr_act);        
  }
//  print_message(ps1, N+1, 3);

  for (i = 0 ; i < N; i++){ // To data
    ii = 0;
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          if ((is[1] == is1[2]) && ((is[0]^is[2]) == is1[1])) {  // check state consistency
            for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
              ic = (&((&(pc[i]))->vector[ci]))->index[0];
              id = ic ^ is[0] ^ is[2];
              if (id == (is1[0] ^ is[2])) { // check codeword consistency
                index =  id;
                vc = (&((&(pc[i]))->vector[ci]))->value;
                vd = vs1 * vc * vs; // update
                ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);
              }
            }
          }
        }
      }
    }
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pc not available
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          if ((is[1] == is1[2]) && ((is[0]^is[2]) == is1[1])) {  // check state consistency
            id = (is1[0] ^ is[2]);
            index =  id;
            vd = vs1 * vs; // update
            ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);
          }
        }
      }
    }
    if (((&(ps1[i+1]))->act_length == 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // ps1 not available
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
        vs = (&((&(ps[i]))->vector[si]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          id = ic ^ is[0] ^ is[2];
          vc = (&((&(pc[i]))->vector[ci]))->value;
          index =  id;
          vd = vc * vs; // update
          ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);          
        }
      }
    }
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // ps not available 
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          id = ic ^ is1[1];
          vc = (&((&(pc[i]))->vector[ci]))->value;
          index =  id;
          vd = vc * vs1; // update
          ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);          
        }
      }
    }
    (&(out[i]))->act_length = ii;
    sort(&(out[i]), temp);
    norm(&(out[i]));  
    select_active(&(out[i]), thr_act);        
  }    
//  print_message(out, N, 1);
  if (out1 != NULL) {
    for (i = 0 ; i < N; i++){ // To codeword
      ii = 0;
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (si = 0; si < (&(ps[i]))->act_length; si++) {
            for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
            vs = (&((&(ps[i]))->vector[si]))->value;
            if ((is[1] == is1[2]) && ((is[0]^is[2]) == is1[1])) {  // check state consistency
              for (di = 0; di < (&(pd[i]))->act_length; di++) {
                id = (&((&(pd[i]))->vector[di]))->index[0];
                if (is1[0]==(is[2]^id)) { // check codeword consistency
                  ic = id ^ is[0] ^ is[2];
                  index =  ic;
                  vd = (&((&(pd[i]))->vector[di]))->value;
                  vc = vs1 * vd * vs; // update
                  ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);
                }
              }
            }
          }
        }
      }
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pd not available
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (si = 0; si < (&(ps[i]))->act_length; si++) {
            for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
            vs = (&((&(ps[i]))->vector[si]))->value;
            if ((is[1] == is1[2]) && ((is[0]^is[2]) == is1[1])) {  // check state consistency
              ic = (is[0] ^ is1[0]);
              index =  ic;
              vc = vs1 * vs; // update
              ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);
            }
          }
        }
      }
      if (((&(ps1[i+1]))->act_length == 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // ps1 not available
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          for (di = 0; di < (&(pd[i]))->act_length; di++) {
            id = (&((&(pd[i]))->vector[di]))->index[0];
            ic = id ^ is[0] ^ is[2];
            vd = (&((&(pd[i]))->vector[di]))->value;
            index =  ic;
            vc = vd * vs; // update
            ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);          
          }
        }
      }
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // ps not available 
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (di = 0; di < (&(pd[i]))->act_length; di++) {
            id = (&((&(pc[i]))->vector[ci]))->index[0];
            vd = (&((&(pd[i]))->vector[di]))->value;
            ic = id ^ is1[1];
            index =  ic;
            vc = vd * vs1; // update
            ii = process_update(ii, ic, ic, &ic, 1, l, thr_act, &(out1[i]), Nc);          
          }
        }
      }
      (&(out1[i]))->act_length = ii;
      sort(&(out1[i]), temp);
      norm(&(out1[i]));  
      select_active(&(out1[i]), thr_act);        
    }    
//    print_message(out1, N, 1);
  }
  clear_message(ps, N+1);
  clear_message(ps1, N+1);
  free(temp); temp = NULL;
  free(l); l = NULL;
}

void update_FSM_sparse_MS(int N, int Ns, int Nd, int Nc, int Aq,  Message *pc, Message *pd, Message *out, Message *out1, float thr_act, int zero_term) {
  /*
   * Description:
   * ------------
   * Method to provide the FSM update for sparsely represented messages. The code is described by
   * the following set of equations (on bitwise-XOR)
   *
   * Forward (to sn+1):
   * 
   * all available:   |  no data       |  no codewords  |  no d no c      |  no sn
   * S0 = s2 + d      |  S0 = c - s0   |  S0 = s2 + d   |  S0 = all       |  S0, S2 = all 
   * S1 = s0 + s2     |  S1 = s0 + s2  |  S1 = s0 + s2  |  S1 = s0 + s2   |  S1 = c - d 
   * S2 = s1          |  S2 = s1       |  S2 = s1       |  S2 = s1        |  
   * c  = s0 + s2 + d |                |                |                 |
   *
   * Backward (to sn):
   * 
   * all available:   |  no data          |  no codewords     |  no d no c      |  no sn+1
   * s0 = S1 - S0 + d |  s0 = c - S0      |  s0 = S1 - S0 +d  |  s0 = S1 + s2   |  s0, s1 = all 
   * s1 = S2          |  s1 = S2          |  s1 = S2          |  s1 = S2        |  s2 = c - d -s0
   * s2 = S0 - d      |  s2 = S0 + S1 -c  |  s2 = S0 - d      |  s2 = all       |  
   * c  = S1 + d      |                   |                   |                 |
   *
   * To data (to d):
   * 
   * all available:   |  no codeword      |  no sn+1          |  no sn
   * d = S0 - s2      |  d = S0 - s2      |  d = c - s0 - s2  |  d = c - S1 
   * d = c - s0 - s2  |  S2 = s1          |                   |  
   * S2 = s1          |  S1 = s0 + s2     |                   |  
   * S1 = s0 + s2     |                   |                   |  
   *
   * To codewords (to c):
   * 
   * all available:   |  no data          |  no sn+1          |  no sn
   * c = d + s0 + s2  |  c = S0 + s0      |  c = d + s0 + s2  |  c = d + S1 
   * S0 = d + s2      |  S2 = s1          |                   |  
   * S2 = s1          |  S1 = s0 + s2     |                   |  
   * S1 = s0 + s2     |                   |                   |  
   *
   * Parameters:
   * -----------
   * N : int
   *   Length of data sequence (including the tail bits to achieve zero final state)
   *
   * Nd : int  
   *   Maximum possible considered data vector length
   *
   * Nc : int  
   *   Maximum possible considered codeword vector length
   *
   * Ns : int  
   *   Maximum possible considered state vector length
   *
   * Aq : int  
   *   Cardinality of data vector (not necessary the same as Nd) 
   *
   * pc : *Message
   *   Sparse message vector representing the codewords
   *
   * pd : *Message
   *   Sparse message vector representing the data
   *
   * out : *Message
   *   Sparse message vector representing for the data output
   *
   * out2 : *Message
   *   Sparse message vector representing for the codeword output (if NULL -> no codeword update is
   *   done)
   *
   * thr_act : float
   *   Probability threshold for considering the message active/inactive 
   *
   * zero_term : int
   *   Flag if the sequence terminates in the zero state (1 -> yes) (otherwise -> no)
   */

  int dim = 3; // Fixed for concrete FSM with constraint length K = 3
  int i, ii, di, ci, si, si1,  j;
  int id, ic;
  long index;
  long Aq2 = Aq*Aq;
  int ic_test;
  int is[3];
  int is1[3];
  float vd, vc, vs, vs1;
  Message *ps1 = alloc_message(Ns, N+1, dim); 
  Message *ps = alloc_message(Ns, N+1, dim); 
  list *l = alloc_list(Ns);
  // Known zero initial  state
  (&((&(ps[0]))->vector[0]))->value = 1;
  for (i = 0; i < dim; i++) (&((&(ps[0]))->vector[0]))->index[i] = 0;
  (&(ps[0]))->act_length = 1;

  if (zero_term == 1) {
    (&((&(ps1[N]))->vector[0]))->value = 1;
    for (i = 0; i < dim; i++) (&((&(ps1[N]))->vector[0]))->index[i] = 0;
    (&(ps1[N]))->act_length = 1;
  }
  else (&(ps1[N]))->act_length = 0;

  Value *temp = (Value *) malloc(sizeof(Value) * Ns);

  for (i = 0 ; i < N; i++){ // Forward
    ii = 0;
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
          vs = (&((&(ps[i]))->vector[si]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = (is[2] + id) % Aq; is1[1] = (is[0] + is[2]) % Aq; is1[2] = is[1]; // Next state
          ic_test = (is[0] + is[2] + id) % Aq;                                // Corresponding codeword
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
            ic = (&((&(pc[i]))->vector[ci]))->index[0];
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
            if (ic == ic_test){ // Find all compatible codewords
              vc = (&((&(pc[i]))->vector[ci]))->value;
              vs1 = vd * vc * vs; // update
              ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);
            }
          }
        }
      }
    }
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pd,ps available, pc unavailable
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
          vs = (&((&(ps[i]))->vector[si]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = (is[2] + id) % Aq; is1[1] = (is[0] + is[2]) % Aq; is1[2] = is[1]; // Next state
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          vs1 = vd * vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length == 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // pc, ps avail, pd unavail
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
        vs = (&((&(ps[i]))->vector[si]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
//          printf("IS:%d %d %d, i:%d, si:%d, ci:%d \n", is[0], is[1],is[2], i, si, ci); 
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          /*  ----------------  FSM description  ----------------------*/
          is1[0] = (Aq + ic - is[0]) % Aq; is1[1] = (is[0] + is[2]) % Aq; is1[2] = is[1]; // Next state
          /*  ---------------------------------------------------------*/
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
          vc = (&((&(pc[i]))->vector[ci]))->value;
          vs1 = vc * vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length == 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // only state available
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
  //        printf("IS:%d %d %d, i:%d, si:%d, di:%d \n", is[0], is[1],is[2], i, si, di); 
        vs = (&((&(ps[i]))->vector[si]))->value;
        /*  ----------------  FSM description  ----------------------*/
        is1[1] = (is[0] + is[2]) % Aq; is1[2] = is[1]; // Next state
        /*  ---------------------------------------------------------*/
        for (si1 = 0; si1 < Aq; si1++) {
          is1[0] = si1; // All possibilities of the state
          index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
          vs1 = vs; // update
          ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
        }
      }
    }
    if (((&(pd[i]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // no state available
      for (di = 0; di < (&(pd[i]))->act_length; di++) {
        id = (&((&(pd[i]))->vector[di]))->index[0];
        vd = (&((&(pd[i]))->vector[di]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          vc = (&((&(pc[i]))->vector[ci]))->value;
          vs1 = vc *vd; // update
          for (is1[0] = 0; is1[0] < Aq; is1[0]++) {
            for (is1[2] = 0; is1[2] < Aq; is1[2]++) {
              /*  ----------------  FSM description  ----------------------*/
              is1[1] = (Aq + ic - id) % Aq;
              /*  ---------------------------------------------------------*/
              index = is1[0] + is1[1] * Aq + is1[2] * Aq2;
              ii = process_update(ii, index, vs1, is1, dim, l, thr_act, &(ps[i+1]), Ns);        
            }
          }
        }
      }
    }
  //    printf("Adjusting active length to %d\n", ii);
    (&(ps[i+1]))->act_length = ii;
    sort(&(ps[i+1]), temp);
    norm(&(ps[i+1]));  
    select_active(&(ps[i+1]), thr_act);        
  }
//  printf("Forward:\n:");print_message(ps, N+1, 3);

  for (i = N ; i > 0; i--){ // Backward
    ii = 0;
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length > 0 )) { // all messages are available
//      printf("C1[%d]\n", i);
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
//          printf("is1:%d %d %d\n", is1[0], is1[1], is1[2]);
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = (Aq + is1[1] - is1[0] + id) % Aq; is[1] = is1[2]; is[2] = (Aq + is1[0] - id) % Aq; // Previous state
          ic_test = (is1[1] + id) % Aq;                                      // Corresponding codeword
          /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
            ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
  //          printf("Testing id:%d, IS:%d %d %d, ic:%d, ic_test:%d\n", id,  is[0], is[1],is[2], ic, ic_test);
            if (ic == ic_test){ // Find all compatible codewords
              vc = (&((&(pc[i-1]))->vector[ci]))->value;
              vs = vd * vc * vs1; // update
              ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
            }
          }
        }
      }
    }
    if (((&(pd[i-1]))->act_length == 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length > 0 )) { // data are not available
//      printf("C2[%d]\n", i);
      for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
        ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
        vc = (&((&(pc[i-1]))->vector[ci]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = (Aq + ic - is1[0]) % Aq ; is[1] = is1[2]; is[2] = (Aq + is1[0] - ic + is1[1]) % Aq; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vc * vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length == 0 ) && ((&(ps1[i]))->act_length > 0 )) { // codewords are not available
//      printf("C3[%d]\n", i);
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i]))->vector[si1]))->value;
          /*  ----------------  FSM description  ----------------------*/
          is[0] = (Aq + is1[1] - is1[0] + id) % Aq; is[1] = is1[2]; is[2] = (Aq + is1[0] - id) % Aq; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vd * vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length == 0 ) && ((&(pc[i-1]))->act_length == 0 ) && ((&(ps1[i]))->act_length > 0 )) { // neither data nor codewords are available
//      printf("C4[%d]\n", i);
      for (si1 = 0; si1 < (&(ps1[i]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i]))->vector[si1]))->index[j];          
        vs1 = (&((&(ps1[i]))->vector[si1]))->value;
        is[1] = is1[2];
        for (is[2] = 0; is[2] < Aq; is[2]++) {
          /*  ----------------  FSM description  ----------------------*/
          is[0] = (is1[1] + is[2]) % Aq; // Previous state
        /*  ---------------------------------------------------------*/
          index = is[0] + is[1] * Aq + is[2] * Aq2;
          vs = vs1; // update
          ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
        }
      }
    }
    if (((&(pd[i-1]))->act_length > 0 ) && ((&(pc[i-1]))->act_length > 0 ) && ((&(ps1[i]))->act_length == 0 )) { // sn+1 not available
//      printf("C5[%d]\n", i);
      for (di = 0; di < (&(pd[i-1]))->act_length; di++) {
        id = (&((&(pd[i-1]))->vector[di]))->index[0];
        vd = (&((&(pd[i-1]))->vector[di]))->value;
        for (ci = 0; ci < (&(pc[i-1]))->act_length; ci++) {
          ic = (&((&(pc[i-1]))->vector[ci]))->index[0];
          vc = (&((&(pc[i-1]))->vector[ci]))->value;
          for (is[1] = 0; is[1] < Aq; is[1]++) {
            for (is[0] = 0; is[0] < Aq; is[0]++) {
              /*  ----------------  FSM description  ----------------------*/
              is[2] = (Aq + Aq +  ic - id - is[0]) % Aq; // Previous state
              /*  ---------------------------------------------------------*/
              index = is[0] + is[1] * Aq + is[2] * Aq2;
              vs = vd * vc; // update
              ii = process_update(ii, index, vs, is, dim, l, thr_act, &(ps1[i-1]), Ns);
            }
          }
        }
      }
    }
    (&(ps1[i-1]))->act_length = ii;
    sort(&(ps1[i-1]), temp);
    norm(&(ps1[i-1]));  
    select_active(&(ps1[i-1]), thr_act);        
  }
//  printf("Backward:\n:");print_message(ps1, N+1, 3);

  for (i = 0 ; i < N; i++){ // To data
    ii = 0;
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          if ((is[1] == is1[2]) && (((is[0]+is[2])%Aq) == is1[1])) {  // check state consistency
            for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
              ic = (&((&(pc[i]))->vector[ci]))->index[0];
              id = (Aq + Aq + ic - is[0] - is[2]) % Aq;
              if (id == ((Aq + is1[0] - is[2])%Aq)) { // check codeword consistency
                index =  id;
                vc = (&((&(pc[i]))->vector[ci]))->value;
                vd = vs1 * vc * vs; // update
                ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);
              }
            }
          }
        }
      }
    }
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pc not available
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          if ((is[1] == is1[2]) && (((is[0]+is[2])%Aq) == is1[1])) {  // check state consistency
            id = (Aq + is1[0] - is[2]) % Aq;
            index =  id;
            vd = vs1 * vs; // update
            ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);
          }
        }
      }
    }
    if (((&(ps1[i+1]))->act_length == 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // ps1 not available
      for (si = 0; si < (&(ps[i]))->act_length; si++) {
        for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
        vs = (&((&(ps[i]))->vector[si]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          id = (Aq + Aq + ic - is[0] - is[2]) % Aq;
          vc = (&((&(pc[i]))->vector[ci]))->value;
          index =  id;
          vd = vc * vs; // update
          ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);          
        }
      }
    }
    if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pc[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // ps not available 
      for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
        for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
        vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
        for (ci = 0; ci < (&(pc[i]))->act_length; ci++) {
          ic = (&((&(pc[i]))->vector[ci]))->index[0];
          id = (Aq + ic - is1[1]) % Aq;
          vc = (&((&(pc[i]))->vector[ci]))->value;
          index =  id;
          vd = vc * vs1; // update
          ii = process_update(ii, id, vd, &id, 1, l, thr_act, &(out[i]), Nd);          
        }
      }
    }
    (&(out[i]))->act_length = ii;
    sort(&(out[i]), temp);
    norm(&(out[i]));  
    select_active(&(out[i]), thr_act);        
  }    
//  print_message(out, N, 1);
  if (out1 != NULL) {
    for (i = 0 ; i < N; i++){ // To codeword
      ii = 0;
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // all messages are available
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (si = 0; si < (&(ps[i]))->act_length; si++) {
            for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
            vs = (&((&(ps[i]))->vector[si]))->value;
            if ((is[1] == is1[2]) && (((is[0]+is[2])%Aq) == is1[1])) {  // check state consistency
              for (di = 0; di < (&(pd[i]))->act_length; di++) {
                id = (&((&(pd[i]))->vector[di]))->index[0];
                if (is1[0]==((is[2]+id)%Aq)) { // check codeword consistency
                  ic = (id + is[0] + is[2]) % Aq;
                  index =  ic;
                  vd = (&((&(pd[i]))->vector[di]))->value;
                  vc = vs1 * vd * vs; // update
                  ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);
                }
              }
            }
          }
        }
      }
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length == 0 ) && ((&(ps[i]))->act_length > 0 )) { // pd not available
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (si = 0; si < (&(ps[i]))->act_length; si++) {
            for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
            vs = (&((&(ps[i]))->vector[si]))->value;
            if ((is[1] == is1[2]) && (((is[0]+is[2])%Aq) == is1[1])) {  // check state consistency
              ic = (is[0] + is1[0]) % Aq;
              index =  ic;
              vc = vs1 * vs; // update
              ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);
            }
          }
        }
      }
      if (((&(ps1[i+1]))->act_length == 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length > 0 )) { // ps1 not available
        for (si = 0; si < (&(ps[i]))->act_length; si++) {
          for (j = 0; j < dim; j++) is[j] = (&((&(ps[i]))->vector[si]))->index[j];
          vs = (&((&(ps[i]))->vector[si]))->value;
          for (di = 0; di < (&(pd[i]))->act_length; di++) {
            id = (&((&(pd[i]))->vector[di]))->index[0];
            ic = (id + is[0] + is[2]) % Aq;
            vd = (&((&(pd[i]))->vector[di]))->value;
            index =  ic;
            vc = vd * vs; // update
            ii = process_update(ii, ic, vc, &ic, 1, l, thr_act, &(out1[i]), Nc);          
          }
        }
      }
      if (((&(ps1[i+1]))->act_length > 0 ) && ((&(pd[i]))->act_length > 0 ) && ((&(ps[i]))->act_length == 0 )) { // ps not available 
        for (si1 = 0; si1 < (&(ps1[i+1]))->act_length; si1++) {
          for (j = 0; j < dim; j++) is1[j] = (&((&(ps1[i+1]))->vector[si1]))->index[j];
          vs1 = (&((&(ps1[i+1]))->vector[si1]))->value;
          for (di = 0; di < (&(pd[i]))->act_length; di++) {
            id = (&((&(pc[i]))->vector[ci]))->index[0];
            vd = (&((&(pd[i]))->vector[di]))->value;
            ic =( id + is1[1]) % Aq;
            index =  ic;
            vc = vd * vs1; // update
            ii = process_update(ii, ic, ic, &ic, 1, l, thr_act, &(out1[i]), Nc);          
          }
        }
      }
      (&(out1[i]))->act_length = ii;
      sort(&(out1[i]), temp);
      norm(&(out1[i]));  
      select_active(&(out1[i]), thr_act);        
    }    
//    print_message(out1, N, 1);
  }
  clear_message(ps, N+1);
  clear_message(ps1, N+1);
  free(temp); temp = NULL;
  free(l); l = NULL;
}

void copy_Message(Message *m1, Message *m2, int dim) {
  /*
   * Copy a message of length 1!
   */ 
  int i; int j;
  m2->act_length =  m1->act_length;
  for (i = 0; i < m1->max_length; i++) {
    (&(m2->vector[i]))->value = (&(m1->vector[i]))->value;
    for (j = 0; j < dim; j++) (&(m2->vector[i]))->index[j] = (&(m1->vector[i]))->index[j];
  }
}


void VN2_update(Message *m1, Message *m2, Message *out, Value *temp) {
  if (m1->act_length == 0) copy_Message(m2, out, 1);
  if (m2->act_length == 0) copy_Message(m1, out, 1);
  if ((m1->act_length > 0) && (m2->act_length > 0)) {
    m1->vector = index_mergesort(m1->vector, m1->act_length, temp);
    m2->vector = index_mergesort(m2->vector, m2->act_length, temp);
    int i1 = 0;
    int i2 = 0;
    int iout = 0;
    while ((i1 < m1->act_length) && (i2 < m2->act_length)) {
      if ((&(m1->vector[i1]))->index[0] > (&(m2->vector[i2]))->index[0]) i2++;
      if ((&(m1->vector[i1]))->index[0] < (&(m2->vector[i2]))->index[0]) i1++;
      if ((&(m1->vector[i1]))->index[0] == (&(m2->vector[i2]))->index[0]) {
          (&(out->vector[iout]))->index[0] = (&(m2->vector[i2]))->index[0];
          (&(out->vector[iout]))->value = (&(m1->vector[i1]))->value * (&(m2->vector[i2]))->value;
          i1++; i2++; iout++;
      }
    }
    out->act_length = iout;
  }
  norm(out);
}

void VN2_update_vec(Message *m1, Message *m2, Message *out, Value *temp, int N) {
  int i;
  for (i = 0; i < N; i++)  VN2_update(&(m1[i]), &(m2[i]), &(out[i]), temp);
}

void interleave(Message *in, Message *out, int len, int *inter) {
  int i; 
  for (i = 0; i < len; i++) copy_Message(&(in[inter[i]]), &(out[i]), 1);
}

void turbo_decode(Message *b, Message *c1, Message *c2, Message *res, int N, int K, int *inter, int *deinter, int Nd, int Nc, int Ns, int Aq, double t) {
  Message *o1 = alloc_message(Nd, N, 1); 
  Message *o2 = alloc_message(Nd, N, 1); 
  Message *i1 = alloc_message(Nd, N, 1); 
  Message *i2 = alloc_message(Nd, N, 1); 
  Message *O1 = alloc_message(Nd, N, 1); 
  Message *O2 = alloc_message(Nd, N, 1); 
  int i, k; int *ind;
  Value *temp = (Value *) malloc(sizeof(Value) * Nd);
  for (i = 0; i < Nd; i++) {
    ind =(int *) malloc(sizeof(int));
    (&(temp[i]))->index = ind;
  }
  for (i = 0; i < N; i++) copy_Message(&(b[i]), &(i1[i]), 1);
  for (k = 0; k < K; k++) {
//    printf("Input mesasage:\n");
//    print_message(i1,N,1);
    update_FSM_sparse(N, Ns, Nd, Nc, Aq, c1, i1, o1, NULL, t, 0);  
//    printf("Output mesasage:\n");
//    print_message(o1,N,1);
    VN2_update_vec(o1, b, O1, temp, N);
    interleave(O1, i2, N, inter);
    update_FSM_sparse(N, Ns, Nd, Nc, Aq, c2, i2, o2, NULL, t, 0);  
    interleave(o2, O2, N, deinter);
    VN2_update_vec(O2, b, i1, temp, N);
  }
//  printf("Turbo output after %d iterations:\n", K);
//  print_message(O2, N, 1);
//  print_message(b, N, 1);
  VN2_update_vec(O2, b, i1, temp, N);
//  printf("Partial result (b,O2):\n");
//  print_message(i1, N, 1);
//  print_message(o1, N, 1);
  VN2_update_vec(i1, o1, res, temp, N);
//  printf("Result:\n");
//  print_message(res, N, 1);

  for (i = 0; i < Nd; i++) free((&(temp[i]))->index);  free(temp);
  clear_message(o1, N);  clear_message(O1, N);  clear_message(i1, N);
  clear_message(o2, N);  clear_message(O2, N);  clear_message(i2, N);
}

void turbo_decode_MS(Message *b, Message *c1, Message *c2, Message *res, int N, int K, int *inter, int *deinter, int Nd, int Nc, int Ns, int Aq, double t) {
  Message *o1 = alloc_message(Nd, N, 1); 
  Message *o2 = alloc_message(Nd, N, 1); 
  Message *i1 = alloc_message(Nd, N, 1); 
  Message *i2 = alloc_message(Nd, N, 1); 
  Message *O1 = alloc_message(Nd, N, 1); 
  Message *O2 = alloc_message(Nd, N, 1); 
  int i, k; int *ind;
  Value *temp = (Value *) malloc(sizeof(Value) * Nd);
  for (i = 0; i < Nd; i++) {
    ind =(int *) malloc(sizeof(int));
    (&(temp[i]))->index = ind;
  }
  for (i = 0; i < N; i++) copy_Message(&(b[i]), &(i1[i]), 1);
  for (k = 0; k < K; k++) {
//    printf("Input mesasage:\n");
//    print_message(i1,N,1);
    update_FSM_sparse_MS(N, Ns, Nd, Nc, Aq, c1, i1, o1, NULL, t, 0);  
//    printf("Output mesasage:\n");
//    print_message(o1,N,1);
    VN2_update_vec(o1, b, O1, temp, N);
    interleave(O1, i2, N, inter);
    update_FSM_sparse_MS(N, Ns, Nd, Nc, Aq, c2, i2, o2, NULL, t, 0);  
    interleave(o2, O2, N, deinter);
    VN2_update_vec(O2, b, i1, temp, N);
  }
//  printf("Turbo output after %d iterations:\n", K);
//  print_message(O2, N, 1);
//  print_message(b, N, 1);
  VN2_update_vec(O2, b, i1, temp, N);
//  printf("Partial result (b,O2):\n");
//  print_message(i1, N, 1);
//  print_message(o1, N, 1);
  VN2_update_vec(i1, o1, res, temp, N);
//  printf("Result:\n");
//  print_message(res, N, 1);

  for (i = 0; i < Nd; i++) free((&(temp[i]))->index);  free(temp);
  clear_message(o1, N);  clear_message(O1, N);  clear_message(i1, N);
  clear_message(o2, N);  clear_message(O2, N);  clear_message(i2, N);
}

void turbo_decode_MS_zero(Message *b, Message *c1, Message *c2, Message *res, int N, int K, int *inter, int *deinter, int Nd, int Nc, int Ns, int Aq, double t) {
  Message *o1 = alloc_message(Nd, N + 3, 1); 
  Message *o2 = alloc_message(Nd, N + 3, 1); 
  Message *i1 = alloc_message(Nd, N + 3, 1); 
  Message *i2 = alloc_message(Nd, N + 3, 1); 
  Message *O1 = alloc_message(Nd, N + 3, 1); 
  Message *O2 = alloc_message(Nd, N + 3, 1); 
  int i, k; int *ind;
  Value *temp = (Value *) malloc(sizeof(Value) * Nd);
  for (i = 0; i < Nd; i++) {
    ind =(int *) malloc(sizeof(int));
    (&(temp[i]))->index = ind;
  }
  for (i = 0; i < N+3; i++) copy_Message(&(b[i]), &(i1[i]), 1);
  for (i = N+3; i < N+6; i++) copy_Message(&(b[i]), &(i2[i-3]), 1);
  for (k = 0; k < K; k++) {
//    printf("Input mesasage:\n");
//    print_message(i1,N,1);
    update_FSM_sparse_MS(N + 3, Ns, Nd, Nc, Aq, c1, i1, o1, NULL, t, 1);  
//    printf("Output mesasage:\n");
//    print_message(o1,N,1);
    VN2_update_vec(o1, b, O1, temp, N);
    interleave(O1, i2, N, inter);
    update_FSM_sparse_MS(N + 3, Ns, Nd, Nc, Aq, c2, i2, o2, NULL, t, 1);  
    interleave(o2, O2, N, deinter);
    VN2_update_vec(O2, b, i1, temp, N);
  }
//  printf("Turbo output after %d iterations:\n", K);
//  print_message(O2, N, 1);
//  print_message(b, N, 1);
  VN2_update_vec(O2, b, i1, temp, N);
//  printf("Partial result (b,O2):\n");
//  print_message(i1, N, 1);
//  print_message(o1, N, 1);
  VN2_update_vec(i1, o1, res, temp, N);
//  printf("Result:\n");
//  print_message(res, N, 1);

  for (i = 0; i < Nd; i++) free((&(temp[i]))->index);  free(temp);
  clear_message(o1, N);  clear_message(O1, N);  clear_message(i1, N);
  clear_message(o2, N);  clear_message(O2, N);  clear_message(i2, N);
}

Message *init_message(int N, int card, double *vec) {
  Message *res = alloc_message(card, N, 1); 
  int i, j;
  Value *temp = (Value *) malloc(sizeof(Value) * card);
  for (i = 0; i < N; i++) {
    if (vec[i * card] >= 0) {
      (&(res[i]))->act_length = card;
      for (j = 0; j < card; j++) {
        (&((&(res[i]))->vector[j]))->index[0] = j;
        (&((&(res[i]))->vector[j]))->value = (float) vec[i * card + j];
      }
      sort(&(res[i]), temp);
      norm(&(res[i]));
      select_active(&(res[i]), 1e-5);
    }
    else (&(res[i]))->act_length = 0;  
  }  
  free(temp); temp = NULL;
  return res;
}

void decide(long *res, Message *b, int N) {
  int i, j; float max; long max_ind;
  for (i = 0; i < N; i++) {
    max = -1; max_ind = -1;
    for (j = 0; j < (&(b[i]))->act_length; j++) {
      if ((&((&(b[i]))->vector[j]))->value > max) {
        max = (&((&(b[i]))->vector[j]))->value;
        max_ind = (&((&(b[i]))->vector[j]))->index[0];
      }
    }
    res[i] = max_ind;
  }
}

void sparse_turbo_decode(double *pc1, double *pc2, double *pd, long *res, long *inter, long *deinter, long K, long N, long Aq, long Nd,long Nc, long Ns, double theta) {
  int i; int j;
/*  printf("Checking import:\nK:%ld, N:%ld, Aq:%ld, Nd:%ld, Nc:%ld, Ns:%ld, theta:%e", K, N, Aq, Nd, Nc, Ns, theta);
  printf("\npc1: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc1[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\npc2: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc2[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nd: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pd[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nres: ");
  for (i = 0; i < N; i++) printf("%ld ", res[i]);
  printf("\ninter: ");
  for (i = 0; i < N; i++) printf("%ld ", inter[i]);
  printf("\ndeinter: ");
  for (i = 0; i < N; i++) printf("%ld ", deinter[i]);*/
  Message *b = init_message(N, Nd, pd);
  Message *c1 = init_message(N, Nc, pc1);
  Message *c2 = init_message(N, Nc, pc2);
  printf("Initialized b:");
  print_message(b,N,1);
  printf("Initialized c1:");
  print_message(c1,N,1);
//  printf("Initialized c2:");
//  print_message(c2,N,1);
  Message *out =  alloc_message(Nd, N, 1);
//  update_FSM_sparse(N, Ns, Nd, Nc, Aq, c1, b, out, NULL, theta, 0);
//  printf("out:");
//  print_message(out,N,1);
  int *interleaver = (int *) malloc(sizeof(int)*N);
  int *deinterleaver = (int *) malloc(sizeof(int)*N);
  for (i = 0; i < N; i++) {interleaver[i] = (int) inter[i]; deinterleaver[i] = (int) deinter[i];}
  turbo_decode(b, c1, c2, out, N, K, interleaver, deinterleaver, Nd, Nc, Ns, Aq, theta);
  decide(res, out, N);
  clear_message(b, N);  clear_message(c1, N);  clear_message(c2, N);
  clear_message(out, N);
  free(interleaver);
  free(deinterleaver);
}

void sparse_turbo_decode_MS(double *pc1, double *pc2, double *pd, long *res, long *inter, long *deinter, long K, long N, long Aq, long Nd,long Nc, long Ns, double theta) {
  int i; int j;
/*  printf("Checking import:\nK:%ld, N:%ld, Aq:%ld, Nd:%ld, Nc:%ld, Ns:%ld, theta:%e", K, N, Aq, Nd, Nc, Ns, theta);
  printf("\npc1: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc1[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\npc2: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc2[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nd: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pd[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nres: ");
  for (i = 0; i < N; i++) printf("%ld ", res[i]);
  printf("\ninter: ");
  for (i = 0; i < N; i++) printf("%ld ", inter[i]);
  printf("\ndeinter: ");
  for (i = 0; i < N; i++) printf("%ld ", deinter[i]);*/
  Message *b = init_message(N, Nd, pd);
  Message *c1 = init_message(N, Nc, pc1);
  Message *c2 = init_message(N, Nc, pc2);
//  printf("Initialized b:");
//  print_message(b,N,1);
//  printf("Initialized c1:");
//  print_message(c1,N,1);
//  printf("Initialized c2:");
//  print_message(c2,N,1);
  Message *out =  alloc_message(Nd, N, 1);
//  update_FSM_sparse_MS(N, Ns, Nd, Nc, Aq, c1, b, out, NULL, theta, 0);
//  printf("out:");
//  print_message(out,N,1);
  int *interleaver = (int *) malloc(sizeof(int)*N);
  int *deinterleaver = (int *) malloc(sizeof(int)*N);
  for (i = 0; i < N; i++) {interleaver[i] = (int) inter[i]; deinterleaver[i] = (int) deinter[i];}
  turbo_decode_MS(b, c1, c2, out, N, K, interleaver, deinterleaver, Nd, Nc, Ns, Aq, theta);
  decide(res, out, N);
  clear_message(b, N);  clear_message(c1, N);  clear_message(c2, N);
  clear_message(out, N);
  free(interleaver);
  free(deinterleaver);
}

void sparse_turbo_decode_MS_zero(double *pc1, double *pc2, double *pd, long *res, long *inter, long *deinter, long K, long N, long Aq, long Nd,long Nc, long Ns, double theta) {
  int i; int j;
/*  printf("Checking import:\nK:%ld, N:%ld, Aq:%ld, Nd:%ld, Nc:%ld, Ns:%ld, theta:%e", K, N, Aq, Nd, Nc, Ns, theta);
  printf("\npc1: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc1[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\npc2: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pc2[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nd: ");
  for (i = 0; i < N; i++) {
    for (j = 0; j < Aq; j++) {
      printf("%.3f ", pd[i*Aq+j]);
    }
    printf("\n");
  }
  printf("\nres: ");
  for (i = 0; i < N; i++) printf("%ld ", res[i]);
  printf("\ninter: ");
  for (i = 0; i < N; i++) printf("%ld ", inter[i]);
  printf("\ndeinter: ");
  for (i = 0; i < N; i++) printf("%ld ", deinter[i]);*/
  Message *b = init_message(N + 6, Nd, pd);
  Message *c1 = init_message(N + 3, Nc, pc1);
  Message *c2 = init_message(N + 3, Nc, pc2);
//  printf("Initialized b:");
//  print_message(b,N,1);
//  printf("Initialized c1:");
//  print_message(c1,N,1);
//  printf("Initialized c2:");
//  print_message(c2,N,1);
  Message *out =  alloc_message(Nd, N, 1);
//  update_FSM_sparse_MS(N, Ns, Nd, Nc, Aq, c1, b, out, NULL, theta, 0);
//  printf("out:");
//  print_message(out,N,1);
  int *interleaver = (int *) malloc(sizeof(int)*N);
  int *deinterleaver = (int *) malloc(sizeof(int)*N);
  for (i = 0; i < N; i++) {interleaver[i] = (int) inter[i]; deinterleaver[i] = (int) deinter[i];}
  turbo_decode_MS_zero(b, c1, c2, out, N, K, interleaver, deinterleaver, Nd, Nc, Ns, Aq, theta);
  decide(res, out, N);
  clear_message(b, N);  clear_message(c1, N);  clear_message(c2, N);
  clear_message(out, N);
  free(interleaver);
  free(deinterleaver);
}
/*
int main() {
  int N = 5; int i; int j;
  int Nc = 4;
  int Nd = 4;
  int Ns = 16;
  long *res = (long *) malloc(sizeof(long)*N);
  Message *pc = alloc_message(Nc, N, 1); 
  Message *pd = alloc_message(Nd, N, 1); 
  Message *out = alloc_message(Nd, N, 1); 
  Message *out1 = alloc_message(Nc, N, 1); 

  for (i = 0; i < N; i++) {
    (&(pd[i]))->act_length = 4;
    (&(pc[i]))->act_length = 4;
    for (j = 0; j < Nd; j++) {
      (&((&(pc[i]))->vector[j]))->index[0] = j;
      (&((&(pd[i]))->vector[j]))->index[0] = j;
    }
  }
  (&((&(pc[0]))->vector[0]))->value = 8.66418153e-02; (&((&(pd[0]))->vector[0]))->value = 6.49159247e-02;
  (&((&(pc[0]))->vector[1]))->value = 7.11628711e-01; (&((&(pd[0]))->vector[1]))->value = 3.42957744e-01;
  (&((&(pc[0]))->vector[2]))->value = 1.70258848e-02; (&((&(pd[0]))->vector[2]))->value = 5.85266939e-02;
  (&((&(pc[0]))->vector[3]))->value = 1.39841351e-01; (&((&(pd[0]))->vector[3]))->value = 3.09202757e-02;
  
  (&(pc[1]))->act_length = 0;
//  (&((&(pc[1]))->vector[0]))->value = 2.50000000e-01; 
  (&((&(pd[1]))->vector[0]))->value = 4.66705965e-01;
//  (&((&(pc[1]))->vector[1]))->value = 2.50000000e-01; 
  (&((&(pd[1]))->vector[1]))->value = 8.10041837e-02;
  //(&((&(pc[1]))->vector[2]))->value = 2.50000000e-01;
  (&((&(pd[1]))->vector[2]))->value = 3.70690301e-02;
  //(&((&(pc[1]))->vector[3]))->value = 2.50000000e-01; 
  (&((&(pd[1]))->vector[3]))->value = 6.43391503e-01;
                                                                                                          
  (&((&(pc[2]))->vector[0]))->value = 1.04493103e-17; (&((&(pd[2]))->vector[0]))->value = 3.11567273e-11;
  (&((&(pc[2]))->vector[1]))->value = 1.23404828e-32; (&((&(pd[2]))->vector[1]))->value = 5.63156485e-01;
  (&((&(pc[2]))->vector[2]))->value = 6.44785951e-01; (&((&(pd[2]))->vector[2]))->value = 3.44132669e-26;
  (&((&(pc[2]))->vector[3]))->value = 7.61482788e-16; (&((&(pd[2]))->vector[3]))->value = 6.22018297e-16;
//  (&(pd[2]))->act_length = 0;

//  (&(pd[3]))->act_length = 0;
//  (&(pc[3]))->act_length = 0;
                                                                                                          
  (&((&(pc[3]))->vector[0]))->value = 3.01662389e-15; (&((&(pd[3]))->vector[0]))->value = 1.53208126e-11;
  (&((&(pc[3]))->vector[1]))->value = 5.29328867e-25; (&((&(pd[3]))->vector[1]))->value = 2.09861346e-21;
  (&((&(pc[3]))->vector[2]))->value = 5.05944809e-01; (&((&(pd[3]))->vector[2]))->value = 3.22137329e-01;
  (&((&(pc[3]))->vector[3]))->value = 8.87784499e-11; (&((&(pd[3]))->vector[3]))->value = 4.41257101e-11;
                                                                                                          
//  (&(pd[0]))->act_length = 0;

  (&((&(pc[4]))->vector[0]))->value = 1.21033882e-28; (&((&(pd[4]))->vector[0]))->value = 6.15253660e-29;
  (&((&(pc[4]))->vector[1]))->value = 3.12500672e-13; (&((&(pd[4]))->vector[1]))->value = 5.64209428e-12;
  (&((&(pc[4]))->vector[2]))->value = 3.20711441e-16; (&((&(pd[4]))->vector[2]))->value = 5.06418168e-18;
  (&((&(pc[4]))->vector[3]))->value = 8.28053592e-01; (&((&(pd[4]))->vector[3]))->value = 4.64403422e-01;
  
//  print_message(pc, N, 1);
//  print_message(pd, N, 1);

  update_FSM_sparse(N, Ns, Nd, Nc, 4, pc, pd, out, out1, 1e-8, 0);

  Value *temp = (Value *) malloc(sizeof(Value) * Nd);
  for (i = 0; i < Nd; i++) (&(temp[i]))->index = (int *) malloc(sizeof(int));
  for (i = 0; i < N; i++) VN2_update(&(pc[i]), &(pd[i]), &(out[i]), temp);
  
  print_message(pd, N, 1);
  print_message(pc, N, 1);
  printf("C update:\n");
  print_message(out, N, 1);

  for (i = 0; i < Nd; i++) free((&(temp[i]))->index);
  free(temp);
  
  int inter [5] = {2, 4, 3, 1, 0};
  interleave(out, pd, 5, inter);
  print_message(pd, N, 1);
  decide(res, pd, N);
  for (i = 0; i < N; i++) printf("%ld ", res[i]);
  printf("\n");
    
//  print_message(out, N, 1);


  clear_message(pc, N);
  clear_message(pd, N);
  clear_message(out, N);
  clear_message(out1, N);

  return 0;
}*/
