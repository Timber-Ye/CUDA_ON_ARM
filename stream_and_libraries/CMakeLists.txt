add_executable(two_streams
        two_streams.cu)

add_executable(single_stream
        single_stream.cu)

add_executable(cublas_gemm
        cublas_gemm.cu)
target_link_libraries(cublas_gemm -lcublas)