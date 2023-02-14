//
// Created by hanchiao on 2/9/23.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include "../matmul/error.cuh"

#define BLOCK_SIZE 32

using namespace cv;

//将RGB图像转化成灰度图
//out = 0.3 * R + 0.59 * G + 0.11 * B
__global__ void im2gray_gpu(uchar3* in_gpu, unsigned char* out_gpu, int imgWidth, int imgHeight){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x < imgWidth && y < imgHeight){
        uchar3 rgb = in_gpu[y * imgWidth + x];

        out_gpu[y * imgWidth + x] = 0.3 * rgb.x + 0.57 * rgb.y + 0.11 * rgb.z;
    }
}


int main(){

    Mat img = imread("data/luna.jpg", ImreadModes::IMREAD_COLOR);
    int imgWidth = img.cols;
    int imgHeight = img.rows;

//    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //申请指针并将它指向GPU空间
    size_t num = imgHeight * imgWidth;
    uchar3* in_gpu;
    unsigned char* out_gpu;
    CHECK(cudaMalloc((void**)&in_gpu,  sizeof(uchar3) * num));
    CHECK(cudaMalloc((void**)&out_gpu, sizeof(unsigned char) * num));

    //将数据从CPU传输到GPU
    CHECK(cudaMemcpy(in_gpu, img.data, sizeof(uchar3) * num, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / BLOCK_SIZE,
                       (imgHeight + threadsPerBlock.y - 1) / BLOCK_SIZE);

    im2gray_gpu<<<blocksPerGrid, threadsPerBlock>>>(in_gpu, out_gpu, imgWidth, imgHeight);

    //将数据从GPU传输到CPU
    CHECK(cudaMemcpy(dst_cpu.data, out_gpu, sizeof(uchar) * num, cudaMemcpyDeviceToHost));

    imwrite("data/luna_gray.png", dst_cpu);

    cudaFree(in_gpu);
    cudaFree(out_gpu);

    return 0;
}
