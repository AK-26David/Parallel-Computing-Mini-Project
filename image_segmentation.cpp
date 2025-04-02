#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>

// ==================== USER CONFIGURATION SECTION ====================
const std::string INPUT_IMAGE_PATH = "input.jpg";
const std::string OUTPUT_IMAGE_PATH = "output.jpg";

// Algorithm parameters
const int NUM_CLUSTERS = 5;        // Number of segmentation clusters
const int MAX_ITERATIONS = 20;     // Maximum K-means iterations
const bool RESIZE_LARGE_IMAGES = true;
const int MAX_IMAGE_DIMENSION = 1024;  // Max dimension for resizing
// =====================================================================

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

// Choose initial centroids using K-means++ algorithm
void initializeCentroids(const cv::Mat& img, std::vector<cv::Vec3f>& centroids, int numClusters) {
    int numPixels = img.rows * img.cols;
    centroids.resize(numClusters);
    std::vector<float> distances(numPixels, std::numeric_limits<float>::max());
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, numPixels - 1);
    
    // Choose first centroid randomly
    int firstIdx = dis(gen);
    centroids[0] = cv::Vec3f(img.at<cv::Vec3b>(firstIdx / img.cols, firstIdx % img.cols));
    
    // Choose remaining centroids
    for (int k = 1; k < numClusters; k++) {
        // Update distances
        float totalDist = 0.0f;
        for (int i = 0; i < numPixels; i++) {
            cv::Vec3b color = img.at<cv::Vec3b>(i / img.cols, i % img.cols);
            
            // Find minimum distance to existing centroids
            float minDist = std::numeric_limits<float>::max();
            for (int c = 0; c < k; c++) {
                float dist = (color[0] - centroids[c][0]) * (color[0] - centroids[c][0]) +
                            (color[1] - centroids[c][1]) * (color[1] - centroids[c][1]) +
                            (color[2] - centroids[c][2]) * (color[2] - centroids[c][2]);
                minDist = std::min(minDist, dist);
            }
            
            distances[i] = minDist;
            totalDist += minDist;
        }
        
        // Choose next centroid with probability proportional to squared distance
        std::uniform_real_distribution<> probDist(0.0, totalDist);
        float threshold = probDist(gen);
        float sum = 0.0f;
        int nextIdx = 0;
        
        for (int i = 0; i < numPixels; i++) {
            sum += distances[i];
            if (sum >= threshold) {
                nextIdx = i;
                break;
            }
        }
        
        centroids[k] = cv::Vec3f(img.at<cv::Vec3b>(nextIdx / img.cols, nextIdx % img.cols));
    }
}

// Main segmentation function
cv::Mat segmentImageWithKMeans(const cv::Mat& inputImg) {
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
    int numPixels = img.rows * img.cols;
    
    // Prepare data for K-means
    cv::Mat samples(numPixels, 3, CV_32F);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y * img.cols + x, z) = img.at<cv::Vec3b>(y, x)[z];
            }
        }
    }
    
    // Run OpenCV's K-means implementation (for comparison)
    std::cout << "Starting OpenCV K-means with " << NUM_CLUSTERS << " clusters..." << std::endl;
    cv::Mat labels, centers;
    cv::kmeans(samples, NUM_CLUSTERS, labels,
              cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 
                              MAX_ITERATIONS, 0.001),
              3, cv::KMEANS_PP_CENTERS, centers);
    
    // Generate colors for visualization
    std::vector<cv::Vec3b> clusterColors = generateDistinctColors(NUM_CLUSTERS);
    
    // Color the segmented image based on cluster assignments
    for (int i = 0; i < numPixels; i++) {
        int cluster = labels.at<int>(i);
        int row = i / img.cols;
        int col = i % img.cols;
        segmented.at<cv::Vec3b>(row, col) = clusterColors[cluster];
    }
    
    // Finish timing and report
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Segmentation completed in " << duration.count() << " ms" << std::endl;
    
    return segmented;
}

int main() {
    try {
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
        cv::Mat segmentedImage = segmentImageWithKMeans(inputImage);
        
        // Save the result
        std::cout << "Saving segmented image to: " << OUTPUT_IMAGE_PATH << std::endl;
        if (!cv::imwrite(OUTPUT_IMAGE_PATH, segmentedImage)) {
            throw std::runtime_error("Error saving output image: " + OUTPUT_IMAGE_PATH);
        }
        
        std::cout << "Segmentation completed successfully!" << std::endl;
        std::cout << "Output saved as: " << OUTPUT_IMAGE_PATH << std::endl;
        
        // Display results
        cv::imshow("Original Image", inputImage);
        cv::imshow("Segmented Image", segmentedImage);
        cv::waitKey(0);
        
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