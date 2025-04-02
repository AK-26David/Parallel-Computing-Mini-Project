#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// OpenCV includes for C
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>

// ==================== USER CONFIGURATION SECTION ====================
// Change these paths to match your image locations
const char* INPUT_IMAGE_PATH = "input.jpg";
const char* OUTPUT_IMAGE_PATH = "output.jpg";

// Algorithm parameters
const int NUM_CLUSTERS = 5;        // Number of segmentation clusters
const int MAX_ITERATIONS = 20;     // Maximum K-means iterations
const int RESIZE_LARGE_IMAGES = 1; // Boolean (1=true, 0=false)
const int MAX_IMAGE_DIMENSION = 1024;  // Max dimension for resizing
// =====================================================================

// CUDA parameters
const int BLOCK_SIZE = 256;

// Error handling macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Define our own uchar3 equivalent struct
typedef struct {
    unsigned char x, y, z;
} pixel3;

// Define float3 equivalent if needed
typedef struct {
    float x, y, z;
} float3_t;

// ==================== CUDA Kernel Definitions ====================

__global__ void assignClustersKernel(const pixel3* pixels, int* labels, const float3_t* centroids, 
                                    int numPixels, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    float minDist = FLT_MAX;
    int bestCluster = 0;
    float3_t pixel = {pixels[idx].x, pixels[idx].y, pixels[idx].z};

    for (int c = 0; c < numClusters; c++) {
        float3_t centroid = centroids[c];
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

__global__ void updateCentroidsKernel(const pixel3* pixels, const int* labels, 
                                    float3_t* newCentroids, int* clusterSizes, 
                                    int numPixels, int numClusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;

    int cluster = labels[idx];
    atomicAdd(&newCentroids[cluster].x, pixels[idx].x);
    atomicAdd(&newCentroids[cluster].y, pixels[idx].y);
    atomicAdd(&newCentroids[cluster].z, pixels[idx].z);
    atomicAdd(&clusterSizes[cluster], 1);
}

__global__ void finalizeCentroidsKernel(float3_t* centroids, const float3_t* newCentroids,
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

// Simple RGB color struct for distinct colors
typedef struct {
    unsigned char b, g, r;
} ColorBGR;

// Generate visually distinct colors for visualization
void generateDistinctColors(ColorBGR* colors, int numColors) {
    for (int i = 0; i < numColors; i++) {
        // Simple hue rotation in HSV space
        float hue = 360.0f * i / numColors;
        float s = 0.9f;
        float v = 0.9f;
        
        float c = v * s;
        float x = c * (1 - fabs(fmod(hue / 60.0f, 2) - 1));
        float m = v - c;
        
        float r, g, b;
        
        if (hue >= 0 && hue < 60) {
            r = c; g = x; b = 0;
        } else if (hue >= 60 && hue < 120) {
            r = x; g = c; b = 0;
        } else if (hue >= 120 && hue < 180) {
            r = 0; g = c; b = x;
        } else if (hue >= 180 && hue < 240) {
            r = 0; g = x; b = c;
        } else if (hue >= 240 && hue < 300) {
            r = x; g = 0; b = c;
        } else {
            r = c; g = 0; b = x;
        }
        
        colors[i].b = (unsigned char)((b + m) * 255);
        colors[i].g = (unsigned char)((g + m) * 255);
        colors[i].r = (unsigned char)((r + m) * 255);
    }
}

// Display CUDA device information
void printDeviceInfo() {
    cudaDeviceProp prop;
    int deviceCount = 0;
    
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("CUDA Devices found: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}

// Choose initial centroids using K-means++ algorithm
void initializeCentroids(const IplImage* img, float3_t* d_centroids, int numClusters) {
    int numPixels = img->height * img->width;
    float3_t* h_centroids = (float3_t*)malloc(numClusters * sizeof(float3_t));
    float* distances = (float*)malloc(numPixels * sizeof(float));
    
    // Initialize distances to maximum
    for (int i = 0; i < numPixels; i++) {
        distances[i] = FLT_MAX;
    }
    
    // Choose first centroid randomly
    int firstIdx = rand() % numPixels;
    int row = firstIdx / img->width;
    int col = firstIdx % img->width;
    
    pixel3* pixel = (pixel3*)(img->imageData + row * img->widthStep) + col;
    h_centroids[0].x = pixel->x;
    h_centroids[0].y = pixel->y;
    h_centroids[0].z = pixel->z;
    
    // Choose remaining centroids
    for (int k = 1; k < numClusters; k++) {
        // Update distances
        float totalDist = 0.0f;
        for (int i = 0; i < img->height; i++) {
            for (int j = 0; j < img->width; j++) {
                int idx = i * img->width + j;
                pixel3* px = (pixel3*)(img->imageData + i * img->widthStep) + j;
                float3_t pxf = {px->x, px->y, px->z};
                
                // Find minimum distance to existing centroids
                float minDist = FLT_MAX;
                for (int c = 0; c < k; c++) {
                    float dist = (pxf.x - h_centroids[c].x) * (pxf.x - h_centroids[c].x) +
                                 (pxf.y - h_centroids[c].y) * (pxf.y - h_centroids[c].y) +
                                 (pxf.z - h_centroids[c].z) * (pxf.z - h_centroids[c].z);
                    if (dist < minDist) {
                        minDist = dist;
                    }
                }
                
                distances[idx] = minDist;
                totalDist += minDist;
            }
        }
        
        // Choose next centroid with probability proportional to squared distance
        float threshold = (float)rand() / RAND_MAX * totalDist;
        float sum = 0.0f;
        int nextIdx = 0;
        
        for (int i = 0; i < numPixels; i++) {
            sum += distances[i];
            if (sum >= threshold) {
                nextIdx = i;
                break;
            }
        }
        
        row = nextIdx / img->width;
        col = nextIdx % img->width;
        pixel = (pixel3*)(img->imageData + row * img->widthStep) + col;
        
        h_centroids[k].x = pixel->x;
        h_centroids[k].y = pixel->y;
        h_centroids[k].z = pixel->z;
    }
    
    // Copy centroids to device
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, numClusters * sizeof(float3_t),
                        cudaMemcpyHostToDevice));
    
    free(h_centroids);
    free(distances);
}

// Main segmentation function
IplImage* segmentImageWithKMeansCUDA(const IplImage* inputImg) {
    // Start timing
    clock_t startTime = clock();
    
    // Prepare image
    IplImage* img = cvCloneImage(inputImg);
    
    // Convert to BGR if needed
    if (img->nChannels == 1) {
        IplImage* colorImg = cvCreateImage(cvGetSize(img), img->depth, 3);
        cvCvtColor(img, colorImg, CV_GRAY2BGR);
        cvReleaseImage(&img);
        img = colorImg;
    } else if (img->nChannels == 4) {
        IplImage* colorImg = cvCreateImage(cvGetSize(img), img->depth, 3);
        cvCvtColor(img, colorImg, CV_BGRA2BGR);
        cvReleaseImage(&img);
        img = colorImg;
    }
    
    // Resize large images if needed
    if (RESIZE_LARGE_IMAGES && 
        (img->width > MAX_IMAGE_DIMENSION || img->height > MAX_IMAGE_DIMENSION)) {
        double scale = fmin((double)MAX_IMAGE_DIMENSION / img->width,
                            (double)MAX_IMAGE_DIMENSION / img->height);
        CvSize newSize = cvSize((int)(img->width * scale), (int)(img->height * scale));
        IplImage* resizedImg = cvCreateImage(newSize, img->depth, img->nChannels);
        cvResize(img, resizedImg, CV_INTER_AREA);
        cvReleaseImage(&img);
        img = resizedImg;
        printf("Image resized to: %dx%d\n", img->width, img->height);
    }
    
    // Output image
    IplImage* segmented = cvCloneImage(img);
    
    // Prepare memory
    int numPixels = img->width * img->height;
    int gridSize = (numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Device pointers
    pixel3* d_pixels = NULL;
    int* d_labels = NULL;
    float3_t* d_centroids = NULL;
    int* d_clusterSizes = NULL;
    float3_t* d_newCentroids = NULL;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_pixels, numPixels * sizeof(pixel3)));
    CUDA_CHECK(cudaMalloc(&d_labels, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_centroids, NUM_CLUSTERS * sizeof(float3_t)));
    CUDA_CHECK(cudaMalloc(&d_clusterSizes, NUM_CLUSTERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, NUM_CLUSTERS * sizeof(float3_t)));
    
    // Prepare data for CUDA
    pixel3* h_pixels = (pixel3*)malloc(numPixels * sizeof(pixel3));
    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            pixel3* pixel = (pixel3*)(img->imageData + i * img->widthStep) + j;
            int idx = i * img->width + j;
            h_pixels[idx] = *pixel;
        }
    }
    
    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_pixels, h_pixels, numPixels * sizeof(pixel3), cudaMemcpyHostToDevice));
    free(h_pixels);
    
    // Initialize centroids
    initializeCentroids(img, d_centroids, NUM_CLUSTERS);
    
    // Run K-means iterations
    float prevInertia = FLT_MAX;
    float currInertia = 0.0f;
    float convergenceThreshold = 0.01f; // 1% change threshold
    
    printf("Starting K-means segmentation with %d clusters...\n", NUM_CLUSTERS);
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Assign each pixel to nearest centroid
        assignClustersKernel<<<gridSize, BLOCK_SIZE>>>(
            d_pixels, d_labels, d_centroids, numPixels, NUM_CLUSTERS
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_newCentroids, 0, NUM_CLUSTERS * sizeof(float3_t)));
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
            printf("Iteration %d/%d\n", iter + 1, MAX_ITERATIONS);
        }
        
        // Check for convergence (optional)
        if (iter > 0 && fabs(currInertia - prevInertia) / prevInertia < convergenceThreshold) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
    }
    
    // Generate colors for visualization
    ColorBGR* clusterColors = (ColorBGR*)malloc(NUM_CLUSTERS * sizeof(ColorBGR));
    generateDistinctColors(clusterColors, NUM_CLUSTERS);
    
    // Copy results back to host
    int* h_labels = (int*)malloc(numPixels * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_labels, d_labels, numPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Color the segmented image based on cluster assignments
    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            int idx = i * img->width + j;
            int cluster = h_labels[idx];
            
            pixel3* pixel = (pixel3*)(segmented->imageData + i * segmented->widthStep) + j;
            pixel->x = clusterColors[cluster].b;
            pixel->y = clusterColors[cluster].g;
            pixel->z = clusterColors[cluster].r;
        }
    }
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_clusterSizes);
    cudaFree(d_newCentroids);
    
    // Free host memory
    free(h_labels);
    free(clusterColors);
    
    // Finish timing and report
    clock_t endTime = clock();
    double duration = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000.0;
    printf("Segmentation completed in %.2f ms\n", duration);
    
    // Release original image
    cvReleaseImage(&img);
    
    return segmented;
}

int main() {
    // Set random seed
    srand((unsigned int)time(NULL));
    
    // Print CUDA device information
    printDeviceInfo();
    
    // Load image from the configured path
    printf("Loading image from: %s\n", INPUT_IMAGE_PATH);
    IplImage* inputImage = cvLoadImage(INPUT_IMAGE_PATH, CV_LOAD_IMAGE_COLOR);
    if (!inputImage) {
        fprintf(stderr, "Error loading image: %s\n", INPUT_IMAGE_PATH);
        return EXIT_FAILURE;
    }
    
    printf("Image loaded successfully. Dimensions: %dx%d, Channels: %d\n", 
            inputImage->width, inputImage->height, inputImage->nChannels);
    
    // Run K-means segmentation
    IplImage* segmentedImage = segmentImageWithKMeansCUDA(inputImage);
    
    // Save the result
    printf("Saving segmented image to: %s\n", OUTPUT_IMAGE_PATH);
    int saveSuccess = cvSaveImage(OUTPUT_IMAGE_PATH, segmentedImage, NULL);
    if (!saveSuccess) {
        fprintf(stderr, "Error saving output image: %s\n", OUTPUT_IMAGE_PATH);
        cvReleaseImage(&inputImage);
        cvReleaseImage(&segmentedImage);
        return EXIT_FAILURE;
    }
    
    printf("Segmentation completed successfully!\n");
    printf("Output saved as: %s\n", OUTPUT_IMAGE_PATH);
    
    // Cleanup
    cvReleaseImage(&inputImage);
    cvReleaseImage(&segmentedImage);
    
    return EXIT_SUCCESS;
}
