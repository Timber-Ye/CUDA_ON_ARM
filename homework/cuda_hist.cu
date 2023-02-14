//
// Created by hanchiao on 2/9/23.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include "../matmul/error.cuh"

#define GrayLevel 256

using namespace std;
using namespace cv;

int h_histogram[GrayLevel] = {0};

__global__ void cal_hist_gpu(unsigned char* in, int* out, int imgHeight, int imgWidth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

//    int index = x + blockDim.x * gridDim.x * y;
    if(x < imgWidth && y < imgHeight){
        int gray_value = in[x + y * imgWidth];
//        int gray_value = in[index];
        atomicAdd(&out[gray_value], 1);
    }
}


void cal_hist_cpu(unsigned char* in, int* out, int imgHeight, int imgWidth)
{
    for(int i=0; i<imgWidth; i++){
        for(int j=0; j<imgHeight; j++){
            out[in[j * imgWidth + i]] ++;
        }
    }
}


int main()
{
    //利用opencv的接口读取图片
    Mat img = imread("data/luna.jpg", ImreadModes::IMREAD_GRAYSCALE);
    int imgWidth = img.cols;
    int imgHeight = img.rows;
//    printf("%d\t %d", imgWidth, imgHeight);

    int *d_histogram;
    CHECK(cudaMalloc((void **)&d_histogram, sizeof(int)*GrayLevel));
    CHECK(cudaMemcpy(d_histogram, h_histogram, sizeof(int)*GrayLevel, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsed_time;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    cal_hist_cpu(img.data, h_histogram, imgHeight, imgWidth);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time 1 using CPU=%g ms.\n", elapsed_time);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    //申请指针并将它指向GPU空间
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char* in_gpu;
    CHECK(cudaMalloc((void**)&in_gpu, num));
    //定义grid和block的维度（形状）
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //将数据从CPU传输到GPU
    CHECK(cudaMemcpy(in_gpu, img.data, num, cudaMemcpyHostToDevice));
    //调用在GPU上运行的核函数
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    cal_hist_gpu<<<blocksPerGrid,threadsPerBlock>>>(in_gpu, d_histogram, imgHeight, imgWidth);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    //将计算结果传回CPU内存
    int *hh_histogram;
    CHECK(cudaMallocHost((void **)&hh_histogram, sizeof(int) * GrayLevel));
    CHECK(cudaMemcpy(hh_histogram, d_histogram, sizeof(int) * GrayLevel, cudaMemcpyDeviceToHost));

    for(int i=0; i<GrayLevel; i++){
        if(h_histogram[i] != hh_histogram[i]){
            printf("validation failed!\n");
            return 0;
        }
    }
    printf("Time 2 using GPU=%g ms.\n", elapsed_time);

    CHECK(cudaFree(d_histogram));
    CHECK(cudaFree(in_gpu));
    CHECK(cudaFreeHost(hh_histogram));
    return 0;
}
