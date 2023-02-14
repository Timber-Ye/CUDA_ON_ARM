//
// Created by hanchiao on 2/9/23.
//
#include <cstdint>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include "../matmul/error.cuh"

#define N 100000
#define TARGET_NUM 10
//#define BLOCKSPERGRID (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK
#define BLOCKSPERGRID 32
#define THREADSPERBLOCK 256

__managed__ int source[N];                                    //input data
__managed__ int final_result_1[TARGET_NUM * BLOCKSPERGRID];   //scalar output
__managed__ int final_result_2[TARGET_NUM];                   //scalar output

__managed__ int final_result_cpu[TARGET_NUM];

double get_time();
void _init(int *ptr, int count);

void print_result(int *result, int count);
__device__ __host__ void insert_value(int* array, int k, int data);
__host__ void _find_topK_cpu(int *ptr, int count, int *result);
__global__ void _find_topK_gpu(int *ptr, int count, int *result);
__device__ __host__ void insert_value(int* array, int k, int data);


int main(){
    //**********************************
    fprintf(stderr, "filling the buffer with %d elements...\n", N);
    _init(source, N);

    //**********************************
    //Now we are going to kick off your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!

    fprintf(stderr, "Running on GPU...\n");

    double t0 = get_time();
    _find_topK_gpu<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(source, N, final_result_1);
    CHECK(cudaGetLastError());  //checking for launch failures
    _find_topK_gpu<<<1, THREADSPERBLOCK>>>(final_result_1, TARGET_NUM * BLOCKSPERGRID, final_result_2);
    CHECK(cudaGetLastError());  //checking for launch failures
    CHECK(cudaDeviceSynchronize()); //checking for run-time failures
    double t1 = get_time();

    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    print_result(final_result_2, TARGET_NUM);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

    double t2 = get_time();
    _find_topK_cpu(source, N, final_result_cpu);
    double t3 = get_time();
//    fprintf(stderr, "CPU max: %u\n", B);

//    ******The last judgement**********
    for(int i=0; i<TARGET_NUM; i++){
        if(final_result_2[i] != final_result_cpu[i]){
            fprintf(stderr, "Test failed!\n");
            exit(-1);
        }
    }

    fprintf(stderr, "Test Passed!\n");


    //****and some timing details*******
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);
    print_result(final_result_cpu, TARGET_NUM);
    return 0;
}


void print_result(int *result, int count){
    printf("[Result]\n");
    for(int i=0; i<count; i++){
        printf("%15d", result[i]);
    }
    printf("\n");
}

__device__ __host__ void insert_value(int* array, int k, int data)
{
    /*
    * 感谢Ken老师提供的函数

    * 作用：判定插入的值是否为目标数组中非重复的最大的k个元素之一，若是，则插入恰当位置（从大到小）并保证其余元素的相对位置不发生改变。

    * array 被插入的目标数组
    * k 目标数组中需要按顺序排列的元素个数
    * data 插入的值
    */
    for (int i = 0; i < k; i++)
    {
        //重复则不需要插入，直接返回
        if (array[i] == data)
        {
            return;
        }
    }
    //因为目标数组最大的k个元素默认再最前面且按照从大到小的顺序排列，所以当插入的值小于第数组中第k个元素即array[k-1]时，即表面插入值并非该数组中最大的k个元素之一。
    if (data < array[k - 1])
        return;
    //经过前面判断，得出插入值符合条件，所以将值插入恰当位置（从大到小）
    for (int i = k - 2; i >= 0; i--)
    {
        if (data > array[i])
            array[i + 1] = array[i];
        else {
            array[i + 1] = data;
            return;
        }
    }
    array[0] = data;
}


__host__ void _find_topK_cpu(int *ptr, int count, int *result)
{
    for(int i=0; i<count;i++){
        insert_value(result, TARGET_NUM, ptr[i]);
    }
}

__global__ void _find_topK_gpu(int *ptr, int count, int *result){
    __shared__ int tile[THREADSPERBLOCK * TARGET_NUM];
    int top_array[TARGET_NUM];

    for (int i = 0; i < TARGET_NUM; i++){
        top_array[i] = INT_MIN;
    }

    for(int idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < count;
        idx += gridDim.x * blockDim.x){
        insert_value(top_array, TARGET_NUM, ptr[idx]);
        //当申请的线程数大于等于输入数据的元素数时将每个线程对应的元素赋值到线程对应的寄存器中；
        // 当申请的线程数小于输入数据的元素数时，利用insert_value函数提取最大的k个值，不足k个的话用依旧使用INT_MIN补全，防止上溢。
    }
    for(int i=0; i<TARGET_NUM; i++){
        tile[TARGET_NUM*threadIdx.x + i] = top_array[i];
    }
    __syncthreads();


    for(int stride = blockDim.x / 2; stride>=1; stride /=2){
        if (threadIdx.x < stride){
            for (int m = 0; m < TARGET_NUM; m++){
                insert_value(top_array, TARGET_NUM, tile[TARGET_NUM * (threadIdx.x + stride) + m]);
            }
        }
        __syncthreads();

        if (threadIdx.x < stride)
        {
            for (int m = 0; m < TARGET_NUM; m++)
            {
                tile[TARGET_NUM* threadIdx.x + m] = top_array[m];
            }
        }
        __syncthreads();
    }

    if (blockIdx.x * blockDim.x < count)
    {
        if (threadIdx.x < TARGET_NUM)
        {
            result[TARGET_NUM * blockIdx.x + threadIdx.x] = tile[threadIdx.x];
        }
    }

}

void _init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}