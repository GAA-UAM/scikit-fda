#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dp_nbhd.h"
#include "dp_grid.h"


void dp_all_edge_weights(
  double *Q1, double *T1, int nsamps1,
  double *Q2, double *T2, int nsamps2,
  int dim,
  double *tv1, int *idxv1, int ntv1,
  double *tv2, int *idxv2, int ntv2,
  double *W, double lam )
{
  int sr, sc;  /* source row and column */
  int tr, tc;  /* target row and column */
  int l1, l2, l3;  /* for multidimensional array mapping */
  int i;

  for ( i=0; i<ntv1*ntv2*ntv1*ntv2; W[i++]=1e6 );

  /* W is a ntv2 x ntv1 x ntv2 x ntv1 array.
   * Weight of edge from (tv1[i],tv2[j]) to (tv1[k],tv2[l])
   * (Cartesian coordinates) is in grid(j,i,l,k).
   * Mapping:
   *  (j,i,l,k) :--> j*ntv1*ntv2*ntv1 +
   *                 i*ntv2*ntv1 +
   *                 l*ntv1 +
   *                 k
   */
  l1 = ntv1 * ntv2 * ntv1;
  l2 = ntv2 * ntv1;
  l3 = ntv1;

  for ( tr=1; tr<ntv2; ++tr )
  {
    for ( tc=1; tc<ntv1; ++tc )
    {
      for ( i=0; i<DP_NBHD_COUNT; ++i )
      {
        sr = tr - dp_nbhd[i][0];
        sc = tc - dp_nbhd[i][1];

        if ( sr < 0 || sc < 0 ) continue;

        /* grid(sr,sc,tr,tc) */
        W[sr*l1+sc*l2+tr*l3+tc] =
         dp_edge_weight( Q1, T1, nsamps1, Q2, T2, nsamps2, dim,
           tv1[sc], tv1[tc], tv2[sr], tv2[tr], idxv1[sc], idxv2[sr], lam );

        /*
        printf( "(%0.2f,%0.2f) --> (%0.2f,%0.2f) = %0.2f\n",
          a, c, b, d, grid[sr*l1+sc*l2+tr*l3+tc] );
        */
      }
    }
  }
}


double dp_costs(
  double *Q1, double *T1, int nsamps1,
  double *Q2, double *T2, int nsamps2,
  int dim,
  double *tv1, int *idxv1, int ntv1,
  double *tv2, int *idxv2, int ntv2,
  double *E, int *P, double lam )
{
  int sr, sc;  /* source row and column */
  int tr, tc;  /* target row and column */
  double w, cand_cost;
  int i;

  E[0] = 0.0;
  for ( i=1; i<ntv1; E[i++]=1e6 );
  for ( i=1; i<ntv2; E[ntv1*i++]=1e6 );

  for ( tr=1; tr<ntv2; ++tr )
  {
    for ( tc=1; tc<ntv1; ++tc )
    {
      E[ntv1*tr + tc] = 1e6;

      for ( i=0; i<DP_NBHD_COUNT; ++i )
      {
        sr = tr - dp_nbhd[i][0];
        sc = tc - dp_nbhd[i][1];

        if ( sr < 0 || sc < 0 ) continue;

        w = dp_edge_weight( Q1, T1, nsamps1, Q2, T2, nsamps2, dim,
          tv1[sc], tv1[tc], tv2[sr], tv2[tr], idxv1[sc], idxv2[sr], lam );

        cand_cost = E[ntv1*sr+sc] + w;
        if ( cand_cost < E[ntv1*tr+tc] )
        {
          E[ntv1*tr+tc] = cand_cost;
          P[ntv1*tr+tc] = ntv1*sr + sc;
        }
      }
    }
  }

  /*
  for ( tr=1; tr<ntv2; ++tr )
  {
    for ( tc=1; tc<ntv1; ++tc )
    {
      printf( "E[%d,%d]=%0.3f, ", tr, tc, E[ntv1*tr+tc] );
      printf( "P[%d,%d]=(%d,%d)\n", tr, tc, P[ntv1*tr+tc]/ntv1,
                                            P[ntv1*tr+tc]%ntv1 );
    }
  }
  */

  return E[ntv1*ntv2-1];
}


double dp_edge_weight(
  double *Q1, double *T1, int nsamps1,
  double *Q2, double *T2, int nsamps2,
  int dim,
  double a, double b,
  double c, double d,
  int aidx, int cidx, double lam)
{
  double res = 0.0;
  int Q1idx, Q2idx;
  int Q1idxnext, Q2idxnext;
  double t1, t2;
  double t1next, t2next;
  double t1nextcand1, t1nextcand2;
  double slope, rslope;
  double dq, dqi;
  int i;

  Q1idx = aidx; /*dp_lookup( T1, nsamps1, a );*/
  Q2idx = cidx; /*dp_lookup( T2, nsamps2, c );*/

  t1 = a;
  t2 = c;

  slope = (d-c)/(b-a);
  rslope = sqrt( slope );

  while( t1 < b && t2 < d )
  {
    if ( Q1idx > nsamps1-2 || Q2idx > nsamps2-2 ) break;

    /* Find endpoint of current interval */
    t1nextcand1 = T1[Q1idx+1];
    t1nextcand2 = a + (T2[Q2idx+1]-c) / slope;

    if ( fabs(t1nextcand1-t1nextcand2) < 1e-6 )
    {
      t1next = T1[Q1idx+1];
      t2next = T2[Q2idx+1];
      Q1idxnext = Q1idx+1;
      Q2idxnext = Q2idx+1;
    } else if ( t1nextcand1 < t1nextcand2 ) {
      t1next = t1nextcand1;
      t2next = c + slope * (t1next - a);
      Q1idxnext = Q1idx+1;
      Q2idxnext = Q2idx;
    } else {
      t1next = t1nextcand2;
      t2next = T2[Q2idx+1];
      Q1idxnext = Q1idx;
      Q2idxnext = Q2idx+1;
    }

    if ( t1next > b ) t1next = b;
    if ( t2next > d ) t2next = d;

    /* Get contribution for current interval */
    dq = 0.0;
    for ( i=0; i<dim; ++i )
    {
      /* Q1 and Q2 are column-major arrays! */
      dqi = Q1[Q1idx*dim+i] - rslope * Q2[Q2idx*dim+i];
      dq += dqi*dqi + lam*(1-rslope)*(1-rslope);
    }
    res += (t1next - t1) * dq;

    t1 = t1next;
    t2 = t2next;
    Q1idx = Q1idxnext;
    Q2idx = Q2idxnext;
  }

  return res;
}


int dp_build_gamma(
  int *P,
  double *tv1, int ntv1,
  double *tv2, int ntv2,
  double *G, double *T )
{
  int sr, sc;
  int tr, tc;
  int p, i;
  int npts;  /* result = length of Tg */

  /* Dry run first, to determine length of Tg */
  npts = 1;
  tr = ntv2-1;
  tc = ntv1-1;
  while( tr > 0 && tc > 0 )
  {
    p = P[tr*ntv1+tc];
    tr = p / ntv1;
    tc = p % ntv1;
    ++npts;
  }

  G[npts-1] = tv2[ntv2-1];
  T[npts-1] = tv1[ntv1-1];

  tr = ntv2-1;
  tc = ntv1-1;
  i = npts-2;
  while( tr > 0 && tc > 0 )
  {
    p = P[tr*ntv1+tc];
    sr = p / ntv1;
    sc = p % ntv1;

    G[i] = tv2[sr];
    T[i] = tv1[sc];

    tr = sr;
    tc = sc;
    --i;
  }

  return npts;
}


int dp_lookup( double *T, int n, double t )
{
  int l, m, r;

  if ( t < T[n-1] )
  {
    l=0;
    r=n;
    m=(l+r)/2;

    /* TODO: are these comparisons OK??? */
    while( 1 )
    {
      if ( t >= T[m+1] )
        l = m;
      else if ( t < T[m] )
        r = m;
      else
        break;

      m = (r+l)/2;
    }

    return m;
  } else {
    return n-2;
  }
}

void dp_all_indexes( double *p, int np, double *tv, int ntv, int *idxv )
{
  int pi;
  int i;

  pi=0;
  for ( i=0; i<ntv; ++i )
  {
    while ( pi < np-2 && tv[i] >= p[pi+1] ) ++pi;
    idxv[i] = pi;
  }
}
