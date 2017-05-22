void update_FSM(int N, double *pc, double *pd, double *out);
void update_FSM_GF4(int N, double *pcv, double *pdv, double *out);
void update_FSM_GFq(int N, double *pcv, double *pdv, double *out, long q);
void update_general_FSM(int N, double *pcv, double *pdv, double *out, long Md, long Ms, long Mc, long *S, long *Q);
void update_general_FSM_both(int N, double *pcv, double *pdv, double *outc, double *outd, long Md, long Ms, long Mc, long *S, long *Q); 
void eff_turbo_update(unsigned int N, float *b, float *c1, float *c2, unsigned int Md, unsigned int Ms, unsigned int Mc, unsigned int *S, unsigned int *Q, unsigned int *inter, unsigned int *deinter, unsigned int K); 
void update_general_fast_FSM(int N, double *pcv, double *pdv, double *out, long Md, long Ms, long Mc, long *S, long *Q, double thresh);
