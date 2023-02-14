//
// Created by hanchiao on 2/9/23.
//

#include <cstdio>
#include "../matmul/error.cuh"
#include <cublas_v2.h>

void print_matrix(int R, int C, double* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)
    {
        for (int c = 0; c < C; ++c)
        {
            printf("%10.6f", A[c * R + r]);
        }
        printf("\n");
    }
}

int main(void){
    int m = 2, n = 2, k = 3;
    int mn = m * n, mk = m * k, nk = n * k;

    double *h_a, *h_b, *h_c;
    CHECK(cudaHostAlloc((void **)&h_a, sizeof(double)*mn, cudaHostAllocDefault)); // 用cudaHostAlloc会出错
    CHECK(cudaHostAlloc((void **)&h_b, sizeof(double)*nk, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&h_c, sizeof(double)*mk, cudaHostAllocDefault));

    for(int i=0; i<mn; i++){
        h_a[i] = i;
    }
    print_matrix(m, n, h_a, "A");

    for(int i=0; i<nk; i++){
        h_b[i] = i;
    }
    print_matrix(n, k, h_b, "B");

    for(int i=0; i<mk; i++){
        h_c[i] = 0;
    }

    double *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, sizeof(double)*mn));
    CHECK(cudaMalloc((void **)&d_b, sizeof(double)*nk));
    CHECK(cudaMalloc((void **)&d_c, sizeof(double)*mk));

    cublasSetVector(mn, sizeof(double), h_a, 1, d_a, 1);
    cublasSetVector(nk, sizeof(double), h_b, 1, d_b, 1);
    cublasSetVector(mk, sizeof(double), h_c, 1, d_c, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, k, n, &alpha, d_a, m, d_b, n, &beta, d_c, m);
    cublasDestroy(handle);
    cublasGetVector(mk, sizeof(double), d_c, 1, h_c, 1);

    print_matrix(m, k, h_c, "C = A x B");

    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}