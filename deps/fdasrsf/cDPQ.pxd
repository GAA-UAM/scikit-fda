cdef extern from "DynamicProgrammingQ2.h":
    void DynamicProgrammingQ2(double *Q1, double *T1, double *Q2, double *T2, int m1, int n1, int n2,
                              double *tv1, double *tv2, int n1v, int n2v, double *G, double *T,
                              double *size, double lam1)