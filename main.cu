#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <string>

// ==================== USER CONFIGURATION SECTION ====================
// Change these paths to match your image locations
const std::string INPUT_IMAGE_PATH = "input.jpg";
const std::string OUTPUT_IMAGE_PATH = "output.jpg";

// Algorithm parameters
const int NUM_CLUSTERS = 5;        // Number of segmentation clusters
const int MAX_ITERATIONS = 20;     // Maximum K-means iterations
const bool RESIZE_LARGE_IMAGES = true;
const int MAX_IMAGE_DIMENSION = 1024;  // Max dimension for resizing
// =====================================================================

// CUDA parameters
const int BLOCK_SIZE = 256;

// Error handling macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ==================== CUDA Kernel Definitions ====================

__global__ void assignClustersKernel(const uchar3* pixels, int* labels, const float3* centroids, 
                                    int numPixels, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    float minDist = FLT_MAX;
    int bestCluster = 0;
    float3 pixel = make_float3(pixels[idx].x, pixels[idx].y, pixels[idx].z);

    for (int c = 0; c < numClusters; c++) {
        float3 centroid = centroids[c];
        float dist = (pixel.x - centroid.x) * (pixel.x - centroid.x) +
                     (pixel.y - centroid.y) * (pixel.y - centroid.y) +
                     (pixel.z - centroid.z) * (pixel.z - centroid.z);
        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    labels[idx] = bestCluster;
}

__global__ void updateCentroidsKernel(const uchar3* pixels, const int* labels, 
                                     float3* newCentroids, int* clusterSizes, 
                                     int numPixels, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    int cluster = labels[idx];
    atomicAdd(&newCentroids[cluster].x, pixels[idx].x);
    atomicAdd(&newCentroids[cluster].y, pixels[idx].y);
    atomicAdd(&newCentroids[cluster].z, pixels[idx].z);
    atomicAdd(&clusterSizes[cluster], 1);
}

__global__ void finalizeCentroidsKernel(float3* centroids, const float3* newCentroids, 
                                       const int* clusterSizes, int numClusters) {
    int idx = threadIdx.x;
    if (idx >= numClusters) return;
    
    if (clusterSizes[idx] > 0) {
        centroids[idx].x = newCentroids[idx].x / clusterSizes[idx];
        centroids[idx].y = newCentroids[idx].y / clusterSizes[idx];
        centroids[idx].z = newCentroids[idx].z / clusterSizes[idx];
    }
}

// ==================== Host Functions ====================

// Generate visually distinct colors for visualization
std::vector<cv::Vec3b> generateDistinctColors(int numColors) {
    std::vector<cv::Vec3b> colors;
    
    for (int i = 0; i < numColors; i++) {
        float hue = 360.0f * i / numColors;
        cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(hue, 0.9f, 0.9f));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        
        float* ptr = rgb.ptr<float>();
        colors.push_back(cv::Vec3b(
            static_cast<uchar>(ptr[0] * 255),
            static_cast<uchar>(ptr[1] * 255),
            static_cast<uchar>(ptr[2] * 255)
        ));
    }
    
    return colors;
}

// Display CUDA device information
void printDeviceInfo() {
    cudaDeviceProp prop;
    int deviceCount = 0;
    
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "CUDA Devices found: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

// Choose initial centroids using K-means++ algorithm
void initializeCentroids(const cv::Mat& img, float3* d_centroids, int numClusters) {
    int numPixels = img.rows * img.cols;
    std::vector<float3> h_centroids(numClusters);
    
    // K-means++ initialization
    std::vector<float> distances(numPixels, std::numeric_limits<float>::max());
    
    // Choose first centroid randomly
    int firstIdx = rand() % numPixels;
    cv::Vec3b firstColor = img.at<cv::Vec3b>(firstIdx / img.cols, firstIdx % img.cols);
    h_centroids[0] = make_float3(firstColor[0], firstColor[1], firstColor[2]);
    
    // Choose remaining centroids
    for (int k = 1; k < numClusters; k++) {
        // Update distances
        float totalDist = 0.0f;
        for (int i = 0; i < numPixels; i++) {
            cv::Vec3b color = img.at<cv::Vec3b>(i / img.cols, i % img.cols);
            float3 pixel = make_float3(color[0], color[1], color[2]);
            
            // Find minimum distance to existing centroids
            float minDist = std::numeric_limits<float>::max();
            for (int c = 0; c < k; c++) {
                float dist = (pixel.x - h_centroids[c].x) * (pixel.x - h_centroids[c].x) +
                             (pixel.y - h_centroids[c].y) * (pixel.y - h_centroids[c].y) +
                             (pixel.z - h_centroids[c].z) * (pixel.z - h_centroids[c].z);
                minDist = std::min(minDist, dist);
            }
            
            distances[i] = minDist;
            totalDist += minDist;
        }
        
        // Choose next centroid with probability proportional to squared distance
        float threshold = static_cast<float>(rand()) / RAND_MAX * totalDist;
        float sum = 0.0f;
        int nextIdx = 0;
        
        for (int i = 0; i < numPixels; i++) {
            sum += distances[i];
            if (sum >= threshold) {
                nextIdx = i;
                break;
            }
        }
        
        cv::Vec3b nextColor = img.at<cv::Vec3b>(nextIdx / img.cols, nextIdx % img.cols);
        h_centroids[k] = make_float3(nextColor[0], nextColor[1], nextColor[2]);
    }
    
    // Copy centroids to device
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), numClusters * sizeof(float3),
                         cudaMemcpyHostToDevice));
}

// Main segmentation function
cv::Mat segmentImageWithKMeansCUDA(const cv::Mat& inputImg) {
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Prepare and check input image
    cv::Mat img;
    if (inputImg.channels() == 1) {
        cv::cvtColor(inputImg, img, cv::COLOR_GRAY2BGR);
    } else if (inputImg.channels() == 4) {
        cv::cvtColor(inputImg, img, cv::COLOR_BGRA2BGR);
    } else {
        img = inputImg.clone();
    }
    
    // Resize large images if needed
    if (RESIZE_LARGE_IMAGES && 
        (img.cols > MAX_IMAGE_DIMENSION || img.rows > MAX_IMAGE_DIMENSION)) {
        double scale = std::min(static_cast<double>(MAX_IMAGE_DIMENSION) / img.cols,
                               static_cast<double>(MAX_IMAGE_DIMENSION) / img.rows);
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
        std::cout << "Image resized to: " << img.cols << "x" << img.rows << std::endl;
    }
    
    // Output image
    cv::Mat segmented = img.clone();
    
    // Prepare memory
    int numPixels = img.rows * img.cols;
    int gridSize = (numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Device pointers
    uchar3* d_pixels = nullptr;
    int* d_labels = nullptr;
    float3* d_centroids = nullptr;
    int* d_clusterSizes = nullptr;
    float3* d_newCentroids = nullptr;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_pixels, numPixels * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&d_labels, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_centroids, NUM_CLUSTERS * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_clusterSizes, NUM_CLUSTERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, NUM_CLUSTERS * sizeof(float3)));
    
    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_pixels, img.data, numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));
    
    // Initialize centroids
    initializeCentroids(img, d_centroids, NUM_CLUSTERS);
    
    // Run K-means iterations
    float prevInertia = std::numeric_limits<float>::max();
    float currInertia = 0.0f;
    float convergenceThreshold = 0.01f; // 1% change threshold
    
    std::cout << "Starting K-means segmentation with " << NUM_CLUSTERS << " clusters..." << std::endl;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Assign each pixel to nearest centroid
        assignClustersKernel<<<gridSize, BLOCK_SIZE>>>(
            d_pixels, d_labels, d_centroids, numPixels, NUM_CLUSTERS
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_newCentroids, 0, NUM_CLUSTERS * sizeof(float3)));
        CUDA_CHECK(cudaMemset(d_clusterSizes, 0, NUM_CLUSTERS * sizeof(int)));
        
        // Update centroids
        updateCentroidsKernel<<<gridSize, BLOCK_SIZE>>>(
            d_pixels, d_labels, d_newCentroids, d_clusterSizes, numPixels, NUM_CLUSTERS
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Finalize centroids
        finalizeCentroidsKernel<<<1, NUM_CLUSTERS>>>(
            d_centroids, d_newCentroids, d_clusterSizes, NUM_CLUSTERS
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Report progress
        if (iter % 5 == 0 || iter == MAX_ITERATIONS - 1) {
            std::cout << "Iteration " << iter + 1 << "/" << MAX_ITERATIONS << std::endl;
        }
        
        // Check for convergence (optional)
        if (iter > 0 && std::abs(currInertia - prevInertia) / prevInertia < convergenceThreshold) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }
    }
    
    // Generate colors for visualization
    std::vector<cv::Vec3b> clusterColors = generateDistinctColors(NUM_CLUSTERS);
    
    // Copy results back to host
    std::vector<int> h_labels(numPixels);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, numPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Color the segmented image based on cluster assignments
    for (int i = 0; i < numPixels; i++) {
        int cluster = h_labels[i];
        int row = i / img.cols;
        int col = i % img.cols;
        segmented.at<cv::Vec3b>(row, col) = clusterColors[cluster];
    }
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_clusterSizes);
    cudaFree(d_newCentroids);
    
    // Finish timing and report
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Segmentation completed in " << duration.count() << " ms" << std::endl;
    
    return segmented;
}

int main() {
    try {
        // Print CUDA device information
        printDeviceInfo();
        
        // Load image from the configured path
        std::cout << "Loading image from: " << INPUT_IMAGE_PATH << std::endl;
        cv::Mat inputImage = cv::imread(INPUT_IMAGE_PATH);
        if (inputImage.empty()) {
            throw std::runtime_error("Error loading image: " + INPUT_IMAGE_PATH);
        }
        
        std::cout << "Image loaded successfully. Dimensions: " 
                  << inputImage.cols << "x" << inputImage.rows << ", "
                  << "Channels: " << inputImage.channels() << std::endl;
        
        // Run K-means segmentation
        cv::Mat segmentedImage = segmentImageWithKMeansCUDA(inputImage);
        
        // Save the result
        std::cout << "Saving segmented image to: " << OUTPUT_IMAGE_PATH << std::endl;
        if (!cv::imwrite(OUTPUT_IMAGE_PATH, segmentedImage)) {
            throw std::runtime_error("Error saving output image: " + OUTPUT_IMAGE_PATH);
        }
        
        std::cout << "Segmentation completed successfully!" << std::endl;
        std::cout << "Output saved as: " << OUTPUT_IMAGE_PATH << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown Exception occurred" << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}