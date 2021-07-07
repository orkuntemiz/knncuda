/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses global memory to store reference and query points.
 *
 * @param ref        refence points
 * @param ref_num     number of reference points
 * @param query      query points
 * @param query_num   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_cuda_global(const float * ref,
                     int           ref_num,
                     const float * query,
                     int           query_num,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index);


/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses texture memory for storing reference points  and memory to store query points.
 *
 * @param ref        refence points
 * @param ref_num     number of reference points
 * @param query      query points
 * @param query_num   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_cuda_texture(const float * ref,
                      int           ref_num,
                      const float * query,
                      int           query_num,
                      int           dim,
                      int           k,
                      float *       knn_dist,
                      int *         knn_index);




