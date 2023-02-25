//
// Created by hanchiao on 2/24/23.
//

#include <cstdio>
#include "../matmul/error.cuh"

#include <vector>

using namespace std;

void TensorCreate(int* tensor, const vector<int>& shape);
int Factorial(const vector<int>& arr, int begin);
void PrintCoordinate(const vector<int>& coordinate);
void TensorExpand(int* new_tensor, const int* tensor, const vector<int>& _target_shape, const vector<int>& _original_shape);
int Indexing(int index, const int* tensor, const vector<int>& _target_shape, const vector<int>& _original_shape);


vector<int>original_shape={1, 2, 1, 3, 6};
vector<int>target_shape={3, 3, 2, 1, 3, 6};
//vector<int>target_shape={1, 2, 1, 3, 6};

void TensorCreate(int* tensor, const int & size){
    for(int i = 0; i < size; i++){
        tensor[i] = i + 1;
    }
}
