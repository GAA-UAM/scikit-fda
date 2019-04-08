
#include "dp_grid.h"

#include <stdlib.h>
#include <math.h>

/* Original code developed by J. Derek Tucker in ElasticFDA.jl. The following
* code is under the MIT license, a copy of the license it is included with it.
*
* 03/25/2019: Modified by Pablo Marcos <pablo.marcosm@estudiante.uam.es>.
*/


#define TOL 1e-6


void dp_optimum_reparam(double* Q1, double* T1, double* Q2, double* T2,
                        int m1, int n1, int n2, double* tv1, double* tv2,
                        int n1v, int n2v, double* G, double* T, double* size,
                        double lam1, int nbhd_dim)
{
    int* idxv1;
    int* idxv2;
    double* E; /* E[ntv1*j+i] = cost of best path to (tv1[i],tv2[j]) */
    int* P; /* P[ntv1*j+i] = predecessor of (tv1[i],tv2[j]) along best path */
    int * dp_nbhd; /* Indexes for grid points */
    int nbhd_count; /* Number of indexes */

    idxv1 = malloc((n1v) * sizeof(*idxv1));
    idxv2 = malloc((n2v) * sizeof(*idxv2));
    E = malloc((n1v) * (n2v) * sizeof(*E));
    P = calloc((n1v) * (n2v), sizeof(*P));

    /* indexes for gridpoints */
    dp_nbhd = dp_generate_nbhd(nbhd_dim, &nbhd_count);


    dp_all_indexes(T1, n1, tv1, n1v, idxv1);
    dp_all_indexes(T2, n2, tv2, n2v, idxv2);

    /* Compute cost of best path from (0,0) to every other grid point */
    dp_costs(Q1, T1, n1, Q2, T2, n2,
             m1, tv1, idxv1, n1v, tv2, idxv2, n2v, E, P, lam1,
             nbhd_count, dp_nbhd);

    /* Reconstruct best path from (0,0) to (1,1) */
    *size = dp_build_gamma(P, tv1, n1v, tv2, n2v, G, T);

    /* free allocated memory */
    free(dp_nbhd);
    free(idxv1);
    free(idxv2);
    free(E);
    free(P);
}


double dp_costs(
    double* Q1, double* T1, int nsamps1,
    double* Q2, double* T2, int nsamps2,
    int dim,
    double* tv1, int* idxv1, int ntv1,
    double* tv2, int* idxv2, int ntv2,
    double* E, int* P, double lam, int nbhd_count, int *dp_nbhd)
{
    int sr, sc;  /* source row and column */
    int tr, tc;  /* target row and column */
    double w, cand_cost;
    int i;

    E[0] = 0.0;
    for (i = 1; i < ntv1; E[i++] = INFINITY);
    for (i = 1; i < ntv2; E[ntv1 * i++] = INFINITY);

    for (tr = 1; tr < ntv2; ++tr) {
        for (tc = 1; tc < ntv1; ++tc) {
            E[ntv1 * tr + tc] = INFINITY;

            for (i = 0; i < 2 * nbhd_count; i += 2) {
                sr = tr - dp_nbhd[i];
                sc = tc - dp_nbhd[i + 1];

                if (sr < 0 || sc < 0) continue;

                w = dp_edge_weight(Q1, T1, nsamps1, Q2, T2, nsamps2, dim,
                                   tv1[sc], tv1[tc], tv2[sr], tv2[tr],
                                   idxv1[sc], idxv2[sr], lam);

                cand_cost = E[ntv1 * sr + sc] + w;
                if (cand_cost < E[ntv1 * tr + tc]) {
                    E[ntv1 * tr + tc] = cand_cost;
                    P[ntv1 * tr + tc] = ntv1 * sr + sc;
                }
            }
        }
    }

    return E[ntv1 * ntv2 - 1];
}


double dp_edge_weight(
    double* Q1, double* T1, int nsamps1,
    double* Q2, double* T2, int nsamps2,
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

    slope = (d - c) / (b - a);
    rslope = sqrt(slope);

    while (t1 < b && t2 < d) {
        if (Q1idx > nsamps1 - 2 || Q2idx > nsamps2 - 2) break;

        /* Find endpoint of current interval */
        t1nextcand1 = T1[Q1idx + 1];
        t1nextcand2 = a + (T2[Q2idx + 1] - c) / slope;

        if (fabs(t1nextcand1 - t1nextcand2) < TOL) {
            t1next = T1[Q1idx + 1];
            t2next = T2[Q2idx + 1];
            Q1idxnext = Q1idx + 1;
            Q2idxnext = Q2idx + 1;
        }
        else if (t1nextcand1 < t1nextcand2) {
            t1next = t1nextcand1;
            t2next = c + slope * (t1next - a);
            Q1idxnext = Q1idx + 1;
            Q2idxnext = Q2idx;
        }
        else {
            t1next = t1nextcand2;
            t2next = T2[Q2idx + 1];
            Q1idxnext = Q1idx;
            Q2idxnext = Q2idx + 1;
        }

        if (t1next > b) t1next = b;
        if (t2next > d) t2next = d;

        /* Get contribution for current interval */
        dq = 0.0;
        for (i = 0; i < dim; ++i) {
            /* Q1 and Q2 are column-major arrays! */
            dqi = Q1[Q1idx * dim + i] - rslope * Q2[Q2idx * dim + i];
            dq += dqi * dqi + lam * (1 - rslope) * (1 - rslope);
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
    int* P,
    double* tv1, int ntv1,
    double* tv2, int ntv2,
    double* G, double* T)
{
    int sr, sc;
    int tr, tc;
    int p, i;
    int npts;  /* result = length of Tg */

    /* Dry run first, to determine length of Tg */
    npts = 1;
    tr = ntv2 - 1;
    tc = ntv1 - 1;
    while (tr > 0 && tc > 0) {
        p = P[tr * ntv1 + tc];
        tr = p / ntv1;
        tc = p % ntv1;
        ++npts;
    }

    G[npts - 1] = tv2[ntv2 - 1];
    T[npts - 1] = tv1[ntv1 - 1];

    tr = ntv2 - 1;
    tc = ntv1 - 1;
    i = npts - 2;
    while (tr > 0 && tc > 0) {
        p = P[tr * ntv1 + tc];
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


int dp_lookup(double* T, int n, double t)
{
    int l, m, r;

    if (t < T[n - 1]) {
        l = 0;
        r = n;
        m = (l + r) / 2;

        while (1) {
            if (t >= T[m + 1])
                l = m;
            else if (t < T[m])
                r = m;
            else
                break;

            m = (r + l) / 2;
        }

        return m;
    }
    else {
        return n - 2;
    }
}

void dp_all_indexes(double* p, int np, double* tv, int ntv, int* idxv)
{
    int i;
    int pi = 0;

    for (i = 0; i < ntv; ++i) {
        while (pi < np - 2 && tv[i] >= p[pi + 1]) ++pi;
        idxv[i] = pi;
    }
}


int gcd(int a, int b) {
    /* Greatest common divisor.
    * Computes the greates common divisor between a and b using the euclids
    * algorithm.
    */

    int temp;

    /* Swap if b > a */
    if(b > a) {
        temp = a;
        a = b;
        b = temp;
    }

    /* Iterative Euclid's algorithm */
    while (b != 0)
    {
        a %= b;
        temp = a;
        a = b;
        b = temp;
    }
    return a;
}

int compute_nbhd_count_rec(int n, int *states) {
    /* Computes the number of elements in the nbhd grid, wich is the number of
    * elements in the set
    *              {(i,j) : gcd(i,j)=1 & 1 <= i,j <= n }
    *
    * This number corresponds with the OEIS A018805 sequence and can be computed
    * using the following formula:
    *
    *               a(n) = n^2 - Sum_{j=2..n} a(floor(n/j))
    */
    int an, j;

    if (states[n] != -1) {
        return states[n];
    }

    an = n * n;

    for(j = 2; j <= n; j++) {
        an -= compute_nbhd_count_rec(n / j, states);
    }

    states[n] = an;

    return an;

}

int compute_nbhd_count(int n) {
    /* Computes the number of elements in the nbhd grid, wich is the number of
    * elements in the set
    *              {(i,j) : gcd(i,j)=1 & 1 <= i,j <= n }
    *
    * This number corresponds with the OEIS A018805 sequence and can be computed
    * using the following formula:
    *
    *               a(n) = n^2 - Sum_{j=2..n} a(floor(n/j))
    */

    int *states;
    int an, i;

    states = malloc((n + 1) * sizeof(*states));
    for(i = 0; i < n + 1; states[i++] = -1);

    an = compute_nbhd_count_rec(n, states);

    free(states);

    return an;
}

int *dp_generate_nbhd(int nbhd_dim, int *nbhd_count) {

    int i, j, k = 0;
    int *dp_nbhd;

    *nbhd_count = compute_nbhd_count(nbhd_dim) ;

    /* Allocate memory for the partition, using the exact amount of we can use
    ~60% of memory that if we use nbhd_dim^2*/
    dp_nbhd = malloc(2 * (*nbhd_count) * sizeof(*dp_nbhd));

    /* dp_nbhd = malloc(2 * nbhd_dim * nbhd_dim * sizeof(*dp_nbhd)); */


    for(i = 1; i <= nbhd_dim; i++) {
        for(j = 1; j <= nbhd_dim; j++) {
            /* If irreducible fraction add as a coordinate */
            if (gcd(i, j) == 1) {
                dp_nbhd[k++] = i;
                dp_nbhd[k++] = j;
            }
        }
    }

    return dp_nbhd;
}
