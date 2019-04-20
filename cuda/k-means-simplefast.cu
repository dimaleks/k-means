#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "helper_math.h"

struct Data {
  Data(int size) : size(size), bytes(size * sizeof(float2)) {
    cudaMalloc(&coordinates, bytes);
    cudaMemset(coordinates, 0, bytes);
  }

  Data(std::vector<float2>& h_coordinates)
  : size(h_coordinates.size()),
    bytes(h_coordinates.size() * sizeof(float2)) {
        
    cudaMalloc(&coordinates, bytes);
    cudaMemcpy(coordinates, h_coordinates.data(), bytes, cudaMemcpyHostToDevice);
  }
  
  
  void clear() {
    cudaMemset(coordinates, 0, bytes);
  }

  ~Data() {
    cudaFree(coordinates);
  }

  float2* coordinates{nullptr};
  int size{0};
  int bytes{0};
};

__device__ inline float
squared_l2_distance(float2 x1, float2 x2) {
  float2 diff = x1-x2;
  return dot(diff, diff);
}

__device__ inline float2
atomicAdd(float2* addr, float2 val) {
    float2 result;
    result.x = atomicAdd(&addr->x, val.x);
    result.y = atomicAdd(&addr->y, val.y);
    return result;
}

__global__ void assign_clusters(const float2* __restrict__ points,
                                int data_size,
                                const float2* __restrict__ means,
                                float2* __restrict__ new_sums,
                                int k,
                                int* __restrict__ counts) {
    
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= data_size) return;
  
  extern __shared__ char shared_memory[];
  
  // First part of the shared memory is for the new centroids coordinates,
  // the second part is for the number of particles assigned
  float2* block_new_sums = (float2*) shared_memory;
  int* block_counts = (int*)(block_new_sums + k);
  
  // Set everything in shared memory to 0
  for (int i=threadIdx.x; i<k; i+=blockDim.x) {
    block_new_sums[i] = {0.0f, 0.0f};
    block_counts[i] = 0;
  }
  
  __syncthreads();

  // Make global loads once.
  const float2 point = points[index];

  // Compute the closest current centroid
  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {

    const float distance = squared_l2_distance(point, means[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  // Add the point coordinate to the NEW centroid coordinate
  // and increment the corresponding points count
  atomicAdd(&block_new_sums[best_cluster], point);
  atomicAdd(&block_counts[best_cluster], 1);
  
  // Wait until all the threads in a block are done updaiting
  __syncthreads();

  
  // Perform global atomics, K operations per block
  if (threadIdx.x < k)
  {
    atomicAdd(&new_sums[threadIdx.x], block_new_sums[threadIdx.x]);
    atomicAdd(&counts[threadIdx.x], block_counts[threadIdx.x]);
  }
}

__global__ void compute_new_means(float2* __restrict__ means,
                                  const float2* __restrict__ new_sum,
                                  const int* __restrict__ counts) {
  const int cluster = threadIdx.x;
  const int count = max(1, counts[cluster]);
  means[cluster] = new_sum[cluster] / count;
}


int main(int argc, const char* argv[]) {
{
  if (argc < 3) {
    std::cerr << "usage: k-means <data-file> <k> [iterations]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto number_of_iterations = (argc == 4) ? std::atoi(argv[3]) : 300;

  std::vector<float2> h_points;
  std::ifstream stream(argv[1]);
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    float x, y;
    uint16_t label;
    line_stream >> x >> y >> label;
    h_points.push_back({x,y});
  }

  const size_t number_of_elements = h_points.size();

  Data d_data(h_points);

  std::mt19937 rng(42);
  std::shuffle(h_points.begin(), h_points.end(), rng);
  std::vector<float2> initial_means{h_points.begin(), h_points.begin() + k};
  Data d_means( initial_means );
  
    
//   for (size_t cluster = 0; cluster < k; ++cluster) {
//     std::cout << initial_means[cluster].x << " " << initial_means[cluster].y << std::endl;
//   }

  const int threads = 64;
  const int blocks = (number_of_elements + threads - 1) / threads;

  //std::cout << "Processing " << number_of_elements << " points on " << blocks
  //          << " blocks x " << threads << " threads" << std::endl;

  // Every block keeps its own centroid data:
  // current x and y sum and number of points (from this block) assigned
  const int shared_memory = k * (sizeof(float2) + sizeof(int));

  Data d_sums(k * blocks);
  int* d_counts;
  cudaMalloc(&d_counts, k * blocks * sizeof(int));
  cudaMemset(d_counts, 0, k * blocks * sizeof(int));

  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      
    cudaMemset(d_counts, 0, k * sizeof(int));
    d_sums.clear();
    
    assign_clusters<<<blocks, threads, shared_memory>>>(d_data.coordinates,
                                                        d_data.size,
                                                        d_means.coordinates,
                                                        d_sums.coordinates,
                                                        k,
                                                        d_counts);
    
    compute_new_means<<<1, k>>>(d_means.coordinates,
                                d_sums.coordinates,
                                d_counts);
  }
  
  cudaDeviceSynchronize();

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cout << "Took: " << duration.count() << "s" << std::endl;

  cudaFree(d_counts);

  std::vector<float2> means(k);
  cudaMemcpy(means.data(), d_means.coordinates, d_means.bytes, cudaMemcpyDeviceToHost);

  for (size_t cluster = 0; cluster < k; ++cluster) {
    std::cout << means[cluster].x << " " << means[cluster].y << std::endl;
  }
}
  cudaDeviceReset();
}
