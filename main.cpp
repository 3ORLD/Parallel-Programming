#include <iostream>
#include <vector>
#include <chrono>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

// Function to load OpenCL kernel from file
string loadKernelSource(const string& filename) {
    ifstream kernel_file(filename);
    if (!kernel_file.is_open()) {
        cerr << "Error opening kernel file: " << filename << endl;
        exit(1);
    }
    stringstream kernel_source;
    kernel_source << kernel_file.rdbuf();
    return kernel_source.str();
}

// Function to create a histogram visualization with better contrast
CImg<unsigned char> createHistogramImage(const std::vector<int>& histogram, int maxHeight = 200) {
    // Find the maximum frequency to normalize the height
    int maxFreq = 1; // Start with 1 to avoid division by zero
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > maxFreq) maxFreq = histogram[i];
    }
    
    std::cout << "Max frequency in histogram: " << maxFreq << std::endl;
    
    // Create an image with a white background for the graph area and some extra space for labels
    CImg<unsigned char> histImg(280, maxHeight + 30, 1, 3, 255); // White background by default
    
    // Colors for drawing (black and white only)
    const unsigned char black[] = {0, 0, 0};
    const unsigned char white[] = {255, 255, 255};
    const unsigned char gray[] = {169, 169, 169}; // Dark gray for grid lines
    
    // Draw a black background underneath the graph for contrast
    histImg.draw_rectangle(10, maxHeight + 10, 270, maxHeight + 30, black, 1.0f); // Black under graph
    
    // Draw the histogram area (white background for the graph itself)
    histImg.draw_rectangle(10, 10, 270, maxHeight + 10, white, 1.0f);  // White for the graph background
    
    // Draw grid lines for better readability (gray grid lines)
    for (int y = 0; y < maxHeight; y += 20) {
        histImg.draw_line(10, maxHeight + 10 - y, 270, maxHeight + 10 - y, gray);
    }
    
    // Draw border (white border around the graph area)
    histImg.draw_rectangle(0, 0, 279, maxHeight + 29, white, 1);
    
    // Draw x and y axis (black lines)
    histImg.draw_line(10, maxHeight + 10, 270, maxHeight + 10, black); // x-axis
    histImg.draw_line(10, 10, 10, maxHeight + 10, black); // y-axis
    
    // Draw histogram bars (black bars for the histogram)
    for (int i = 0; i < 256; i++) {
        int barHeight = static_cast<int>((static_cast<double>(histogram[i]) / maxFreq) * maxHeight);
        if (barHeight > 0) {
            // Draw rectangle instead of line for each bin to make it thicker (black bars)
            histImg.draw_rectangle(10 + i, maxHeight + 10 - barHeight, 
                                   11 + i, maxHeight + 10, 
                                   black, 1.0f);
        }
    }
    
    // Add labels for x-axis (black text)
    histImg.draw_text(5, maxHeight + 20, "0", black);
    histImg.draw_text(125, maxHeight + 20, "128", black);
    histImg.draw_text(255, maxHeight + 20, "255", black);
    
    return histImg;
}

// Function to print help message
void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform (index starting from 0)" << std::endl;
    std::cerr << "  -d : select device (index starting from 0)" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    string image_filename = "test.pgm"; // Ensure this image exists in the same folder
    int selected_platform = -1;
    int selected_device = -1;
    bool list_devices = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "-h") {
            print_help();
            return 0;
        }
        if (string(argv[i]) == "-l") {
            list_devices = true;
        }
        if (string(argv[i]) == "-p" && i + 1 < argc) {
            selected_platform = stoi(argv[++i]);
        }
        if (string(argv[i]) == "-d" && i + 1 < argc) {
            selected_device = stoi(argv[++i]);
        }
    }

    if (list_devices) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return 1;
        }

        for (size_t i = 0; i < platforms.size(); ++i) {
            std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (size_t j = 0; j < devices.size(); ++j) {
                std::cout << "  Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
            }
        }
        return 0;
    }

    try {
        // Load the image using CImg
        CImg<unsigned char> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        int width = image_input.width();
        int height = image_input.height();
        int total_pixels = width * height;
        const int NUM_BINS = 256; // 8-bit grayscale

        // Initialize OpenCL - create context manually
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return 1;
        }

        // Select platform
        cl::Platform selected_platform_obj;
        if (selected_platform >= 0 && selected_platform < platforms.size()) {
            selected_platform_obj = platforms[selected_platform];
        } else {
            selected_platform_obj = platforms[0]; // Default to first platform
        }

        std::cout << "Selected platform: " << selected_platform_obj.getInfo<CL_PLATFORM_NAME>() << std::endl;

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(selected_platform_obj)(),
            0
        };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        // Get devices
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            std::cerr << "No OpenCL devices found" << std::endl;
            return 1;
        }

        // Select device
        cl::Device selected_device_obj;
        if (selected_device >= 0 && selected_device < devices.size()) {
            selected_device_obj = devices[selected_device];
        } else {
            selected_device_obj = devices[0]; // Default to first device
        }

        std::cout << "Selected device: " << selected_device_obj.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Create command queue with profiling enabled
        cl::CommandQueue queue(context, selected_device_obj, CL_QUEUE_PROFILING_ENABLE);
        std::cout << "Command queue created." << std::endl;

        // Extract raw image data (assuming grayscale)
        std::vector<unsigned char> h_inputImage(total_pixels);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                h_inputImage[y * width + x] = image_input(x, y, 0);
            }
        }

        // Create OpenCL buffers
        cl::Buffer d_inputImage(context, CL_MEM_READ_ONLY, total_pixels * sizeof(unsigned char));
        cl::Buffer d_outputImage(context, CL_MEM_WRITE_ONLY, total_pixels * sizeof(unsigned char));
        cl::Buffer d_histogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
        cl::Buffer d_cumulativeHistogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
        cl::Buffer d_lut(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));

        // Vector to store host-side results
        std::vector<int> h_histogram(NUM_BINS, 0);
        std::vector<int> h_cumulativeHistogram(NUM_BINS, 0);
        std::vector<int> h_lut(NUM_BINS, 0);
        std::vector<unsigned char> h_outputImage(total_pixels);

        // Load OpenCL kernel source
        string kernelSource = loadKernelSource("kernels/my_kernels.cl");

        // Create program from source
        cl::Program program(context, kernelSource);
        
        try {
            // Build the program
            program.build(devices);
        } catch (const cl::Error& e) {
            std::cerr << "OpenCL build error: " << e.what() << std::endl;
            std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device_obj) << std::endl;
            return 1;
        }

        // Create kernels
        cl::Kernel histogramKernel(program, "calculateHistogram");
        cl::Kernel scanKernel(program, "prefixSum");
        cl::Kernel lutKernel(program, "normalizeLUT");
        cl::Kernel applyLUTKernel(program, "applyLUT");

        // Copy input image to device
        queue.enqueueWriteBuffer(d_inputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned char), h_inputImage.data());

        // Initialize histogram buffer with zeros
        queue.enqueueFillBuffer(d_histogram, 0, 0, NUM_BINS * sizeof(int));

        // STEP 1: Calculate histogram
        histogramKernel.setArg(0, d_inputImage);
        histogramKernel.setArg(1, d_histogram);
        histogramKernel.setArg(2, total_pixels);

        // Launch histogram kernel with appropriate work size - use multiple of 64 for better performance
        cl::Event histogramEvent;
        size_t workGroupSize = 64;
        size_t globalWorkSize = ((total_pixels + workGroupSize - 1) / workGroupSize) * workGroupSize;
        queue.enqueueNDRangeKernel(
            histogramKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),
            cl::NDRange(workGroupSize),
            nullptr,
            &histogramEvent
        );
        
        // Wait for histogram calculation to complete
        histogramEvent.wait();
        
        // Read back histogram for display and verification
        queue.enqueueReadBuffer(d_histogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_histogram.data());
        
        // Debug: Print some histogram values to verify data
        std::cout << "First few histogram values: ";
        for (int i = 0; i < 10; i++) {
            std::cout << h_histogram[i] << " ";
        }
        std::cout << std::endl;
        
        // Check if histogram has valid data
        int histogramSum = 0;
        for (int i = 0; i < NUM_BINS; i++) {
            histogramSum += h_histogram[i];
        }
        std::cout << "Histogram sum: " << histogramSum << " (should be close to total_pixels: " << total_pixels << ")" << std::endl;
        
        // Create a visual representation of the input histogram
        CImg<unsigned char> inputHistImg = createHistogramImage(h_histogram, 200);
        CImgDisplay disp_inputHist(inputHistImg, "Input Image Histogram");

        // STEP 2: Calculate cumulative histogram (prefix sum)
        scanKernel.setArg(0, d_histogram);
        scanKernel.setArg(1, d_cumulativeHistogram);
        scanKernel.setArg(2, NUM_BINS);

        cl::Event scanEvent;
        queue.enqueueNDRangeKernel(
            scanKernel,
            cl::NullRange,
            cl::NDRange(NUM_BINS),  // Process all histogram bins
            cl::NDRange(std::min(256, NUM_BINS)),  // Work group size
            nullptr,
            &scanEvent
        );
        
        // Wait for scan to complete
        scanEvent.wait();
        
        // Read back cumulative histogram for verification
        queue.enqueueReadBuffer(d_cumulativeHistogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_cumulativeHistogram.data());
        
        // Debug: Check cumulative histogram values
        std::cout << "First few cumulative histogram values: ";
        for (int i = 0; i < 10; i++) {
            std::cout << h_cumulativeHistogram[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Last cumulative value: " << h_cumulativeHistogram[NUM_BINS-1] 
                  << " (should be close to total_pixels: " << total_pixels << ")" << std::endl;

        // STEP 3: Calculate lookup table for histogram equalization
        lutKernel.setArg(0, d_cumulativeHistogram);
        lutKernel.setArg(1, d_lut);
        lutKernel.setArg(2, total_pixels);
        lutKernel.setArg(3, NUM_BINS);

        cl::Event lutEvent;
        queue.enqueueNDRangeKernel(
            lutKernel,
            cl::NullRange,
            cl::NDRange(NUM_BINS),
            cl::NullRange,
            nullptr,
            &lutEvent
        );
        
        // Wait for LUT calculation to complete
        lutEvent.wait();
        
        // Read back LUT for verification
        queue.enqueueReadBuffer(d_lut, CL_TRUE, 0, NUM_BINS * sizeof(int), h_lut.data());
        
        // Debug: Check LUT values
        std::cout << "First few LUT values: ";
        for (int i = 0; i < 10; i++) {
            std::cout << h_lut[i] << " ";
        }
        std::cout << std::endl;

        // STEP 4: Apply LUT to input image to create equalized output
        applyLUTKernel.setArg(0, d_inputImage);
        applyLUTKernel.setArg(1, d_lut);
        applyLUTKernel.setArg(2, d_outputImage);
        applyLUTKernel.setArg(3, total_pixels);

        cl::Event applyEvent;
        queue.enqueueNDRangeKernel(
            applyLUTKernel,
            cl::NullRange,
            cl::NDRange(globalWorkSize),  // Process all pixels
            cl::NDRange(workGroupSize),
            nullptr,
            &applyEvent
        );
        
        // Wait for LUT application to complete
        applyEvent.wait();

        // Read back the result
        queue.enqueueReadBuffer(d_outputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned char), h_outputImage.data());

        // Create output image
        CImg<unsigned char> image_output(width, height, 1, 1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image_output(x, y, 0) = h_outputImage[y * width + x];
            }
        }

        // Display output image
        CImgDisplay disp_output(image_output, "Equalized Image");

        // Create a visual representation of the normalized cumulative histogram (LUT) in black and white
        CImg<unsigned char> norm_cum_hist_image = createHistogramImage(h_lut, 200);
        CImgDisplay disp_norm_cum_hist(norm_cum_hist_image, "Normalized Cumulative LUT");

        // Create a visual representation of the cumulative histogram in black and white
        CImg<unsigned char> cum_hist_image = createHistogramImage(h_cumulativeHistogram, 200);
        CImgDisplay disp_cum_hist(cum_hist_image, "Cumulative Histogram");

        // Print timing information
        cl_ulong histogramTime = histogramEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                histogramEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        cl_ulong scanTime = scanEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                           scanEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        cl_ulong lutTime = lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                          lutEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        cl_ulong applyTime = applyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                            applyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();

        std::cout << "Timing results:" << std::endl;
        std::cout << "  Histogram calculation: " << histogramTime / 1000000.0 << " ms" << std::endl;
        std::cout << "  Prefix scan: " << scanTime / 1000000.0 << " ms" << std::endl;
        std::cout << "  LUT generation: " << lutTime / 1000000.0 << " ms" << std::endl;
        std::cout << "  LUT application: " << applyTime / 1000000.0 << " ms" << std::endl;
        std::cout << "  Total kernel execution: " << (histogramTime + scanTime + lutTime + applyTime) / 1000000.0 << " ms" << std::endl;

        // Wait for user to close the display windows
        while (!disp_input.is_closed() && !disp_output.is_closed() && 
               !disp_inputHist.is_closed() && !disp_norm_cum_hist.is_closed() && 
               !disp_cum_hist.is_closed()) {
            CImgDisplay::wait(disp_input, disp_output, disp_inputHist, disp_norm_cum_hist, disp_cum_hist);
        }
        
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
    
}
