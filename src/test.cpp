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
/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
int main(void) {

    // Parameters
    const int ref_num = 16384;
    const int query_num = 4096;
    const int dim = 128;
    const int k = 16;

    // Display
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n", ref_num);
    printf("- Number query points     : %d\n", query_num);
    printf("- Dimension of points     : %d\n", dim);
    printf("- Number of neighbors     : %d\n\n", k);

    // Sanity check
    if (ref_num < k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float* ref = (float*)malloc(ref_num * dim * sizeof(float));
    float* query = (float*)malloc(query_num * dim * sizeof(float));
    float* knn_dist = (float*)malloc(query_num * k * sizeof(float));
    int* knn_index = (int*)malloc(query_num * k * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n");
        free(ref);
        free(query);
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_num, query, query_num, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    printf("Ground truth computation in progress...\n\n");
    if (!knn_c(ref, ref_num, query, query_num, dim, k, knn_dist, knn_index)) {
        free(ref);
        free(query);
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    // Test all k-NN functions
    printf("TESTS\n");
    test(ref, ref_num, query, query_num, dim, k, knn_dist, knn_index, &knn_c, "knn_c", 2);
    test(ref, ref_num, query, query_num, dim, k, knn_dist, knn_index, &knn_cuda_global, "knn_cuda_global", 100);
    test(ref, ref_num, query, query_num, dim, k, knn_dist, knn_index, &knn_cuda_texture, "knn_cuda_texture", 100);

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
