add_executable(sobel
        sobel.cu)
target_link_libraries(sobel ${OpenCV_LIBS})

add_executable(im2gray
        im2gray.cu)
target_link_libraries(im2gray ${OpenCV_LIBS})

add_executable(find_max
        find_max.cu)

add_executable(find_top10
        find_top10.cu)

add_executable(test
        test.cu)

add_executable(matrix_trans
        matrix_t.cu)

add_executable(cuda_hist
        cuda_hist.cu)
target_link_libraries(cuda_hist ${OpenCV_LIBS})
