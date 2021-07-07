
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <cstdio>

/**
 * Initializes randomly the reference and query points. For test purposes (this will replaced by given data)
 *
 * @param ref        refence points
 * @param ref_num     number of reference points
 * @param query      query points
 * @param query_num   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float* ref,
    int     ref_num,
    float*  query,
    int     query_num,
    int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i = 0; i < ref_num * dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i = 0; i < query_num * dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}




/**
 * Computes the Euclidean distance between a reference point and a query point.
 * Iterates through all the dimensions of the data points, sum the differences for each dimensions then take the square root.
 * CPU implementation of computing distances, it calculates the point wise euqlidian distance.
 * 
 * @param ref          refence points
 * @param ref_num       number of reference points
 * @param query        query points
 * @param query_num     number of query points
 * @param dim          dimension of points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
float compute_distance(const float* ref,
    int           ref_num,
    const float*  query,
    int           query_num,
    int           dim,
    int           ref_index,
    int           query_index) {
    float sum = 0.f; 
    // Loop through all the dimensions of the point
    for (int d = 0; d < dim; ++d) {
        const float diff = ref[d * ref_num + ref_index] - query[d * query_num + query_index]; // Locate the location of the query and reference points for each dimension
        sum += diff * diff;
    }
    return sqrtf(sum); //float version of squared root operation
}



#define BLOCK_DIM 16 // For shared memory allocation


/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 * GPU implementation of computing distances, we do this operation for a multiple points. Instead of getting two points like in CPU implementation we will get two matrices, which is reference and query matrices.
 * 
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch (padding) of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch (padding) of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
__global__ void compute_distances(float* ref,
    int     ref_width,
    int     ref_pitch,
    float* query,
    int     query_width,
    int     query_pitch,
    int     height,
    float* dist) {

    
    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float A_s[BLOCK_DIM][BLOCK_DIM];
    __shared__ float B_s[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int start_A;
    __shared__ int start_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f; // Sum of squared distances

    // The pitch is then the number of elements allocated for a single row, including the extra bytes (padding bytes).
    // Loop parameters
    start_A = BLOCK_DIM * blockIdx.y; // shared memory matrix for reference
    start_B = BLOCK_DIM * blockIdx.x; // shared memory matrix for query
    step_A = BLOCK_DIM * ref_pitch; // Step Size
    step_B = BLOCK_DIM * query_pitch; // Step Size
    end_A = start_A + (height - 1) * ref_pitch; // Last  row of the sub matrice A

    // Conditions which defines the boundry of the operation.
    int cond0 = (start_A + tx < ref_width); // used to write in shared memory
    int cond1 = (start_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
    int cond2 = (start_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix. Assumed that ref > query.
    for (int a = start_A, b = start_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a / ref_pitch + ty < height) {
            A_s[ty][tx] = (cond0) ? ref[a + ref_pitch * ty + tx] : 0; // in ref matrix, go down pitch times * ty, go right tx times load that element
            B_s[ty][tx] = (cond1) ? query[b + query_pitch * ty + tx] : 0; // in query matrix, go down pitch times * ty, go right tx times load that element
        }
        else {
            A_s[ty][tx] = 0;
            B_s[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix, all the combinations in the sub matrice is calculated
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k) {
                float tmp = A_s[k][ty] - B_s[k][tx];
                ssd += tmp * tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[(start_A + ty) * query_pitch + start_B + tx] = ssd; // unique index for a distance array, means that:
        // for one reference point in a reference matrix which is determined by: (start_A + ty) * query_pitch
        // for one query point in a query matrix which is determined by start_B + tx
        // distance array is formed between every query and reference point
    }
}

/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the texture memory
 * @param ref_width    number of reference points
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
__global__ void compute_distance_texture(cudaTextureObject_t ref,
    int                 ref_width,
    float* query,
    int                 query_width,
    int                 query_pitch,
    int                 height,
    float* dist) { // not to have confusion the matrices are formed as [dimensions, num_of_points] = [height, width] it may be formed vise-versa
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < query_width && idy < ref_width) {
        float ssd = 0.f;
        for (int i = 0; i < height; i++) {
            float tmp = tex2D<float>(ref, (float)idy, (float)i) - query[i * query_pitch + idx];
            ssd += tmp * tmp;
        }
        dist[idy * query_pitch + idx] = ssd;
    }
} 

/**
 * Gathers at the beginning of the `dist` array the k smallest values and their
 * respective index (in the initial array) in the `index` array. After this call,
 * only the k-smallest distances are available. All other distances might be lost.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 * The best case input is an array that is already sorted. In this case insertion sort has a linear running time (i.e., O(n)).
 * During each iteration, the first remaining element of the input is only compared with the right-most element of the sorted subsection of the array.
 * The simplest worst case input is an array sorted in reverse order. The set of all worst case inputs consists of all arrays where each element is the smallest or second-smallest of the elements before it. 
 * In these cases every iteration of the inner loop will scan and shift the entire sorted subsection of the array before inserting the next element. This gives insertion sort a quadratic running time (i.e., O(n2)).
 * The average case is also quadratic, which makes insertion sort impractical for sorting large arrays. 
 * However, insertion sort is one of the fastest algorithms for sorting very small arrays, even faster than quicksort; 
 * indeed, good quicksort implementations use insertion sort for arrays smaller than a certain threshold
 * In this case we did not sort the whole array, since then the worst case scnerio does not hold, the complexity will behave like O(n). Since we find the minimum of unsorted array, and insert it to the beggining.
 * Due to difference mentioned above, this algorithm can be called alternated insertion sort.
 * 
 * @param dist    array containing the `length` distances
 * @param index   array containing the index of the k smallest distances
 * @param length  total number of distances
 * @param k       number of smallest distances to locate
 */
void  insertion_sort_cpu(float* dist, int* index, int length, int k) {

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i = 1; i < length; ++i) {

        // Store current distance and associated index
        float curr_dist = dist[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th already sorted smallest value
        if (i >= k && curr_dist >= dist[k - 1]) {
            continue;
        }
        // Take the min and insert it to the beginning of the array, then sort rest of the array.
        // Shift values (and indexes) higher that the current distance to the right
        int j = std::min(i, k - 1);
        while (j > 0 && dist[j - 1] > curr_dist) {
            dist[j] = dist[j - 1];
            index[j] = index[j - 1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j] = curr_dist;
        index[j] = curr_index;
    }
}

/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 * The best case input is an array that is already sorted. In this case insertion sort has a linear running time (i.e., O(n)).
 * During each iteration, the first remaining element of the input is only compared with the right-most element of the sorted subsection of the array.
 * The simplest worst case input is an array sorted in reverse order. The set of all worst case inputs consists of all arrays where each element is the smallest or second-smallest of the elements before it. 
 * In these cases every iteration of the inner loop will scan and shift the entire sorted subsection of the array before inserting the next element. This gives insertion sort a quadratic running time (i.e., O(n2)).
 * The average case is also quadratic, which makes insertion sort impractical for sorting large arrays. 
 * However, insertion sort is one of the fastest algorithms for sorting very small arrays, even faster than quicksort; 
 * indeed, good quicksort implementations use insertion sort for arrays smaller than a certain threshold
 * In this case we did not sort the whole array, since then the worst case scnerio does not hold, the complexity will behave like O(n). Since we find the minimum of unsorted array, and insert it to the beggining.
 * Due to difference mentioned above, this algorithm can be called alternated insertion sort.
 * 
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
__global__ void insertion_sort_cuda(float* dist,
    int     dist_pitch,
    int* index,
    int     index_pitch,
    int     width,
    int     height,
    int     k) {

    // Column position // Since we read from 1D array 1D indexing will be enough
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (idx < width) {

        // Pointer shift
        float* p_dist = dist + idx;
        int* p_index = index + idx;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i = 1; i < height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i * dist_pitch];
            int   curr_index = i;

            // Skip the current value if its index is >= k and if it's higher the k-th already sorted smallest value
            if (i >= k && curr_dist >= p_dist[(k - 1) * dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k - 1);
            while (j > 0 && p_dist[(j - 1) * dist_pitch] > curr_dist) {
                p_dist[j * dist_pitch] = p_dist[(j - 1) * dist_pitch];
                p_index[j * index_pitch] = p_index[(j - 1) * index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j * dist_pitch] = curr_dist;
            p_index[j * index_pitch] = curr_index;
        }
    }
}


/**
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
 */
__global__ void compute_sqrt(float* dist, int width, int pitch, int k) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < k)
        dist[idy * pitch + idx] = sqrt(dist[idy * pitch + idx]);
    // take the square root of the distance matrix and replace the distance matrix with it
}


/*
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * If the function executed correctly it will return true.
 * 
 * @param ref        refence points
 * @param ref_num     number of reference points
 * @param query      query points
 * @param query_num   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_num x k distances
 * @param knn_index  output array containing the query_num x k indexes
 */
bool knn_c(const float* ref,
    int           ref_num,
    const float* query,
    int           query_num,
    int           dim,
    int           k,
    float* knn_dist,
    int* knn_index) {

    // Allocate local array to store all the distances / indexes for a given query point 
    // CPU memory allocation
    float* dist = (float*)malloc(ref_num * sizeof(float));
    int* index = (int*)malloc(ref_num * sizeof(int));

    // Allocation checks
    if (!dist || !index) {
        printf("Memory allocation error\n");
        free(dist);
        free(index);
        return false;
    }

    // Process one query point at the time
    // naive CPU algorithm of KNN
    for (int i = 0; i < query_num; ++i) {

        // Compute all distances / indexes
        for (int j = 0; j < ref_num; ++j) {
            dist[j] = compute_distance(ref, ref_num, query, query_num, dim, j, i);
            index[j] = j;
        }

        // Sort distances / indexes 
        insertion_sort_cpu(dist, index, ref_num, k);

        // Copy k smallest distances and their associated index
        for (int j = 0; j < k; ++j) {
            knn_dist[j * query_num + i] = dist[j];
            knn_index[j * query_num + i] = index[j];
        }
    }

    // Memory clean-up
    free(dist);
    free(index);

    return true;

}

bool knn_cuda_global(const float* ref,
    int           ref_num,
    const float* query,
    int           query_num,
    int           dim,
    int           k,
    float* knn_dist,
    int* knn_index) {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int = sizeof(int);

    // Return variables
    cudaError_t err0, err1, err2, err3;

    // Allocate global memory
    float* ref_dev = NULL;
    float* query_dev = NULL;
    float* dist_dev = NULL;
    int* index_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    /**
    * cudaMallocPitch is used other than cudaMalloc. The reason of that is:
    * The number of memory access operation it will take depends on the number of memory words this row takes. The number of bytes in a memory word depends on the implementation in our case query and reference points.
    * To minimize the number of memory accesses when reading a single row, we must assure that we start the row on the start of a word, hence we must pad the memory for every row until the start of a new one.
    * Since we usually want to treat each row in parallel, we can ensure that we can access it simulateously by padding each row to the start of a new bank.
    * Long story short instead of allocating the 2D array with cudaMalloc, we will use cudaMallocPitch which is a best practice
    *
    * Note that the pitch here is the return value of the function: cudaMallocPitch checks what it should be on your system and returns the appropriate value. What cudaMallocPitch does is the following:
    *
    * 1)Allocate the first row.
    * 2)Check if the number of bytes allocated makes it correctly aligned. For example that it is a multiple of 128.
    * 3)If not, allocate further bytes to reach the next multiple of 128. the pitch is then the number of bytes allocated for a single row, including the extra bytes (padding bytes).
    * 4)Reiterate for each row.
    * At the end, we have typically allocated more memory than necessary because each row is now the size of pitch
    *
    * Formal definition of cudaMallocPitch:
    * Allocates pitched memory on the device
    *
    * Allocates at least \p width (in bytes) * \p height bytes of linear memory
    * on the device and returns in \p *devPtr a pointer to the allocated memory.
    * The function may pad the allocation to ensure that corresponding pointers
    * in any given row will continue to meet the alignment requirements for
    * coalescing as the address is updated from row to row. The pitch returned in
    * \p *pitch by ::cudaMallocPitch() is the width in bytes of the allocation.
    * The intended usage of \p pitch is as a separate parameter of the allocation,
    * used to compute addresses within the 2D array. Given the row and column of
    * an array element of type \p T, the address is computed as:
    * \code
       T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
      \endcode
    *
    * For allocations of 2D arrays, it is recommended that programmers consider
    * performing pitch allocations using ::cudaMallocPitch(). Due to pitch
    * alignment restrictions in the hardware, this is especially true if the
    * application will be performing 2D memory copies between different regions
    * of device memory (whether linear memory or CUDA arrays).
    */
    err0 = cudaMallocPitch((void**)&ref_dev, &ref_pitch_in_bytes, ref_num * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_num * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&dist_dev, &dist_pitch_in_bytes, query_num * size_of_float, ref_num);
    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_num * size_of_int, k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Deduce pitch values, find the padding etc
    size_t ref_pitch = ref_pitch_in_bytes / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch = dist_pitch_in_bytes / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Copy reference and query data from the host to the device, if cudaMallocPitch is used cudaMemcpy2D should be used.
    /** If we use cudaMemcpy, we will copy all the memory allocated with cudaMallocPitch, including the padded bytes between each rows. What we must do to avoid padding memory is copying each row one by one.
    * Or we can tell the CUDA API that we want only the useful memory from the memory we allocated with padding bytes for its convenience so if it could deal with its own mess automatically it would be very nice indeed, thank you. 
    * And here enters cudaMemcpy2D
    */

    err0 = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref, ref_num * size_of_float, ref_num * size_of_float, dim, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_num * size_of_float, query_num * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Compute the squared Euclidean distances
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0(query_num / BLOCK_DIM, ref_num / BLOCK_DIM, 1);
    // round up the grid sizes
    if (query_num % BLOCK_DIM != 0) {
        grid0.x += 1;
    }
    if (ref_num % BLOCK_DIM != 0) {
        grid0.y += 1;
    }
    compute_distances << <grid0, block0 >> > (ref_dev, ref_num, ref_pitch, query_dev, query_num, query_pitch, dim, dist_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Sort the distances with their respective indexes
    // distance matrix has 16 x 16 = 256 elements due to shared memory size. Also distance matrix is 1D array, therefore 256 x 1 thread count is enough. 
    // Also in the sorting algorithm we deal with 1D indexing.
    dim3 block1(256, 1, 1);
    dim3 grid1(query_num / 256, 1, 1);
    if (query_num % 256 != 0) {
        grid1.x += 1;
    }
    insertion_sort_cuda << <grid1, block1 >> > (dist_dev, dist_pitch, index_dev, index_pitch, query_num, ref_num, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Compute the square root of the k smallest distances
    // since our shared memory size is 16, the shared memory matrices are 16 x 16, therefore 16 x 16 thread count is enough for reading and writing them from the global memory.
    // Also, in square root kernel we deal with 2D indexing, in order to reconstruct the matrice
    dim3 block2(16, 16, 1); 
    dim3 grid2(query_num / 16, k / 16, 1);
    if (query_num % 16 != 0) {
        grid2.x += 1;
    }
    if (k % 16 != 0) {
        grid2.y += 1;
    }
    compute_sqrt << <grid2, block2 >> > (dist_dev, query_num, query_pitch, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    err0 = cudaMemcpy2D(knn_dist, query_num * size_of_float, dist_dev, dist_pitch_in_bytes, query_num * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_num * size_of_int, index_dev, index_pitch_in_bytes, query_num * size_of_int, k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Memory clean-up
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);

    return true;
}

bool knn_cuda_texture(const float* ref,
    int           ref_num,
    const float* query,
    int           query_num,
    int           dim,
    int           k,
    float* knn_dist,
    int* knn_index) {

    // Constants
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int = sizeof(int);

    // Return variables
    cudaError_t err0, err1, err2;


    // Allocate global memory
    float* query_dev = NULL;
    float* dist_dev = NULL;
    int* index_dev = NULL;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    /**
    * cudaMallocPitch is used other than cudaMalloc. The reason of that is:
    * The number of memory access operation it will take depends on the number of memory words this row takes. The number of bytes in a memory word depends on the implementation in our case query and reference points.
    * To minimize the number of memory accesses when reading a single row, we must assure that we start the row on the start of a word, hence we must pad the memory for every row until the start of a new one.
    * Since we usually want to treat each row in parallel, we can ensure that we can access it simulateously by padding each row to the start of a new bank.
    * Long story short instead of allocating the 2D array with cudaMalloc, we will use cudaMallocPitch which is a best practice
    * 
    * Note that the pitch here is the return value of the function: cudaMallocPitch checks what it should be on your system and returns the appropriate value. What cudaMallocPitch does is the following:
    *
    * 1)Allocate the first row.
    * 2)Check if the number of bytes allocated makes it correctly aligned. For example that it is a multiple of 128.
    * 3)If not, allocate further bytes to reach the next multiple of 128. the pitch is then the number of bytes allocated for a single row, including the extra bytes (padding bytes).
    * 4)Reiterate for each row.
    * At the end, we have typically allocated more memory than necessary because each row is now the size of pitch
    * 
    * Formal definition of cudaMallocPitch:
    * Allocates pitched memory on the device
    *
    * Allocates at least \p width (in bytes) * \p height bytes of linear memory
    * on the device and returns in \p *devPtr a pointer to the allocated memory.
    * The function may pad the allocation to ensure that corresponding pointers
    * in any given row will continue to meet the alignment requirements for
    * coalescing as the address is updated from row to row. The pitch returned in
    * \p *pitch by ::cudaMallocPitch() is the width in bytes of the allocation.
    * The intended usage of \p pitch is as a separate parameter of the allocation,
    * used to compute addresses within the 2D array. Given the row and column of
    * an array element of type \p T, the address is computed as:
    * \code
       T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
      \endcode
    *
    * For allocations of 2D arrays, it is recommended that programmers consider
    * performing pitch allocations using ::cudaMallocPitch(). Due to pitch
    * alignment restrictions in the hardware, this is especially true if the
    * application will be performing 2D memory copies between different regions
    * of device memory (whether linear memory or CUDA arrays).
    */

    // Reference object will be a texture object which will handled seperately, below is the same as the shared / global memory implementation
    err0 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_num * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&dist_dev, &dist_pitch_in_bytes, query_num * size_of_float, ref_num);
    err2 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_num * size_of_int, k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("ERROR: Memory allocation error (cudaMallocPitch)\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Deduce pitch values
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch = dist_pitch_in_bytes / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Copy query data from the host to the device
    err0 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_num * size_of_float, query_num * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Allocate CUDA array for reference points
    // cudaMallocArray for allocating memory for texture object
    cudaArray* ref_array_dev = NULL;
    // 32 bits being appropriate for float data type. That is why we pass 32 as the x parameter of cudaCreateChannelDesc
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    err0 = cudaMallocArray(&ref_array_dev, &channel_desc, ref_num, dim);
    if (err0 != cudaSuccess) {
        printf("ERROR: Memory allocation error (cudaMallocArray)\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }

    // Copy reference points from host to device
    err0 = cudaMemcpyToArray(ref_array_dev, 0, 0, ref, ref_num * size_of_float * dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        return false;
    }

    // Resource descriptor
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = ref_array_dev;

    // Texture descriptor
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    // Create the texture
    cudaTextureObject_t ref_tex_dev = 0;
    err0 = cudaCreateTextureObject(&ref_tex_dev, &res_desc, &tex_desc, NULL);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to create the texture\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        return false;
    }
    // Kernel executions are same as the global cuda implementation
    // Compute the squared Euclidean distances
    dim3 block0(16, 16, 1);
    dim3 grid0(query_num / 16, ref_num / 16, 1);
    if (query_num % 16 != 0) {
        grid0.x += 1;
    }
    if (ref_num % 16 != 0) grid0.y += 1;
    compute_distance_texture << <grid0, block0 >> > (ref_tex_dev, ref_num, query_dev, query_num, query_pitch, dim, dist_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }

    // Sort the distances with their respective indexes
    dim3 block1(256, 1, 1);
    dim3 grid1(query_num / 256, 1, 1);
    if (query_num % 256 != 0) grid1.x += 1;
    insertion_sort_cuda << <grid1, block1 >> > (dist_dev, dist_pitch, index_dev, index_pitch, query_num, ref_num, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }

    // Compute the square root of the k smallest distances
    dim3 block2(16, 16, 1);
    dim3 grid2(query_num / 16, k / 16, 1);
    if (query_num % 16 != 0) grid2.x += 1;
    if (k % 16 != 0)        grid2.y += 1;
    compute_sqrt << <grid2, block2 >> > (dist_dev, query_num, query_pitch, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    err0 = cudaMemcpy2D(knn_dist, query_num * size_of_float, dist_dev, dist_pitch_in_bytes, query_num * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_num * size_of_int, index_dev, index_pitch_in_bytes, query_num * size_of_int, k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }

    // Memory clean-up
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFreeArray(ref_array_dev);
    cudaDestroyTextureObject(ref_tex_dev);

    return true;
}
