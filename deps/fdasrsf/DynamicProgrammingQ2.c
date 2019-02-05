#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dp_grid.h"

void DynamicProgrammingQ2(double *Q1, double *T1, double *Q2, double *T2, int m1, int n1, int n2,
double *tv1, double *tv2, int n1v, int n2v, double *G, double *T, double *size, double lam1){
  int *idxv1 = 0;
  int *idxv2 = 0;
  double *E = 0; /* E[ntv1*j+i] = cost of best path to (tv1[i],tv2[j]) */
  int *P = 0; /* P[ntv1*j+i] = predecessor of (tv1[i],tv2[j]) along best path */

  idxv1=(int*)malloc((n1v)*sizeof(int));
  idxv2=(int*)malloc((n2v)*sizeof(int));
  E=(double*)malloc((n1v)*(n2v)*sizeof(double));
  P=(int*)calloc((n1v)*(n2v),sizeof(int));

  /* dp_costs() needs indexes for gridpoints precomputed */
  dp_all_indexes( T1, n1, tv1, n1v, idxv1 );
  dp_all_indexes( T2, n2, tv2, n2v, idxv2 );

  /* Compute cost of best path from (0,0) to every other grid point */
  dp_costs( Q1, T1, n1, Q2, T2, n2,
    m1, tv1, idxv1, n1v, tv2, idxv2, n2v, E, P, lam1 );

  /* Reconstruct best path from (0,0) to (1,1) */
  *size = dp_build_gamma( P, tv1, n1v, tv2, n2v, G, T );

  // free allocated memory
  free(idxv1); free(idxv2); free(E); free(P);
}
