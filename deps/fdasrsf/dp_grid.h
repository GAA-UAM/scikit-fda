#ifndef DP_GRID_H
#define DP_GRID_H 1

/**
 * Computes weights of all edges in the DP matching graph, 
 * which is needed for the Floyd-Warshall all-pairs shortest-path 
 * algorithm.
 *
 * The matrix of edge weights E must be allocated ahead of time by 
 * the caller.  E is an ntv1 x ntv2 x ntv1 x ntv2 matrix.  If i and k 
 * are column indices and j and l are row indices, then the weight of the 
 * edge from gridpoint (tv1[i],tv2[j]) to gridpoint (tv1[k],tv2[l]) is 
 * stored in E(j,i,l,k) when this function returns.
 * Mapping: 
 *  (j,i,l,k) :--> j*ntv1*ntv2*ntv1 + 
 *                 i*ntv2*ntv1 +
 *                 l*ntv1 + 
 *                 k
 *
 * \param Q1 values of the first SRVF
 * \param T1 changepoint parameters of the first SRVF
 * \param nsamps1 the length of T1
 * \param Q2 values of the second SRVF
 * \param T2 changepoint parameters of the second SRVF
 * \param nsamps2 the length of T2
 * \param dim dimension of the ambient space
 * \param tv1 the Q1 (column) parameter values for the DP grid
 * \param idxv1 Q1 indexes for tv1, as computed by \c dp_all_indexes()
 * \param ntv1 the length of tv1
 * \param tv2 the Q2 (row) parameter values for the DP grid
 * \param idxv2 Q2 indexes for tv2, as computed by \c dp_all_indexes()
 * \param ntv2 the length of tv2
 * \param E [output] pointer to the edge weight matrix.  Must already be 
 *          allocated, with size (ntv1*ntv2)^2.
 */
void dp_all_edge_weights( 
  double *Q1, double *T1, int nsamps1,
  double *Q2, double *T2, int nsamps2,
  int dim, 
  double *tv1, int *idxv1, int ntv1, 
  double *tv2, int *idxv2, int ntv2, 
  double *W, double lam );

/**
 * Computes cost of best path from (0,0) to all other gridpoints.
 *
 * \param Q1 values of the first SRVF
 * \param T1 changepoint parameters of the first SRVF
 * \param nsamps1 the length of T1
 * \param Q2 values of the second SRVF
 * \param T2 changepoint parameters of the second SRVF
 * \param nsamps2 the length of T2
 * \param dim dimension of the ambient space
 * \param tv1 the Q1 (column) parameter values for the DP grid
 * \param idxv1 Q1 indexes for tv1, as computed by \c dp_all_indexes()
 * \param ntv1 the length of tv1
 * \param tv2 the Q2 (row) parameter values for the DP grid
 * \param idxv2 Q2 indexes for tv2, as computed by \c dp_all_indexes()
 * \param ntv2 the length of tv2
 * \param E [output] on return, E[ntv2*i+j] holds the cost of the best 
 *        path from (0,0) to (tv1[i],tv2[j]) in the grid.
 * \param P [output] on return, P[ntv2*i+j] holds the predecessor of 
 *        (tv1[i],tv2[j]).  If predecessor is (tv1[k],tv2[l]), then 
 *        P[ntv2*i+j] = k*ntv2+l.
 * \return E[ntv1*ntv2-1], the cost of the best path from (tv1[0],tv2[0]) 
 *         to (tv1[ntv1-1],tv2[ntv2-1]).
 */
double dp_costs(
  double *Q1, double *T1, int nsamps1, 
  double *Q2, double *T2, int nsamps2,
  int dim, 
  double *tv1, int *idxv1, int ntv1, 
  double *tv2, int *idxv2, int ntv2, 
  double *E, int *P, double lam );

/**
 * Computes the weight of the edge from (a,c) to (b,d) in the DP grid.
 *
 * \param Q1 values of the first SRVF
 * \param T1 changepoint parameters of the first SRVF
 * \param nsamps1 the length of T1
 * \param Q2 values of the second SRVF
 * \param T2 changepoint parameters of the second SRVF
 * \param nsamps2 the length of T2
 * \param dim dimension of the ambient space
 * \param a source Q1 parameter
 * \param b target Q1 parameter
 * \param c source Q2 parameter
 * \param d target Q2 parameter
 * \param aidx index such that Q1[aidx] <= a < Q1[aidx+1]
 * \param cidx index such that Q2[cidx] <= c < Q2[cidx+1]
 */
  double dp_edge_weight(
  double *Q1, double *T1, int nsamps1, 
  double *Q2, double *T2, int nsamps2,
  int dim,
  double a, double b, 
  double c, double d, 
  int aidx, int cidx, double lam );
  

/**
 * Given predecessor table P, builds the piecewise-linear reparametrization 
 * function gamma.
 *
 * G and T must already be allocated with size max(ntv1,ntv2).  The actual 
 * number of points on gamma will be the return value.
 *
 * \param P P[ntv2*i+j] holds the predecessor of (tv1[i],tv2[j]).  If 
 *        predecessor is (tv1[k],tv2[l]), then P[ntv2*i+j] = k*ntv2+l.
 * \param tv1 the Q1 (column) parameter values for the DP grid
 * \param ntv1 the length of tv1
 * \param tv2 the Q2 (row) parameter values for the DP grid
 * \param ntv2 the length of tv2
 * \param G [output] reparametrization function values
 * \param T [output] reparametrization changepoint parameters
 * \return the length of G (same as length of T).
 */
int dp_build_gamma( 
  int *P, 
  double *tv1, int ntv1, 
  double *tv2, int ntv2,
  double *G, double *T );

/**
 * Given t in [0,1], return the integer i such that t lies in the interval 
 * [T[i],T[i+1]) (or returns n-2 if t==T[n-1]).
 *
 * \param T an increasing sequence
 * \param n the length of T
 * \param t the parameter value to lookup ( T[0] <= t <= T[n-1] ).
 * \return the integer i such that t lies in the interval [T[i],T[i+1]) 
 *         (or n-2 if t==T[n-1]).
 */
int dp_lookup( double *T, int n, double t );

/**
 * 1-D table lookup for a sorted array of query points.
 *
 * Given a partition p and an increasing sequence of numbers tv between 
 * p[0] and p[np-1], computes the sequence of indexes idxv such that for 
 * i=0,...,ntv-1, p[idxv[i]] <= tv[i] < p[idxv[i]+1].  If tv[i]==p[np-1], 
 * then idxv[i] will be set to np-2.
 *
 * \param p an increasing sequence (the table)
 * \param np the length of \a p
 * \param tv an increasing sequence (the query points)
 * \param ntv the length of \a tv
 * \param idxv [output] pre-allocated array of \a ntv ints to hold result
 */
void dp_all_indexes( double *p, int np, double *tv, int ntv, int *idxv );

#endif /* DP_GRID_H */
