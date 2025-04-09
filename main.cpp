#include <iostream>
#include <vector>
#include <chrono>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

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

CImg<unsigned char> createHistogramImage(const std::vector<int>& histogram, int maxHeight = 200) {
    int maxFreq = 1;
    for (size_t i = 0; i < histogram.size(); i++) {
        if (histogram[i] > maxFreq) maxFreq = histogram[i];
    }
    
    CImg<unsigned char> histImg(280, maxHeight + 30, 1, 3, 255);
    const unsigned char black[] = {0, 0, 0};
    const unsigned char white[] = {255, 255, 255};
    const unsigned char gray[] = {169, 169, 169};
    
    histImg.draw_rectangle(10, maxHeight + 10, 270, maxHeight + 30, black, 1.0f);
    histImg.draw_rectangle(10, 10, 270, maxHeight + 10, white, 1.0f);
    
    for (int y = 0; y < maxHeight; y += 20) {
        histImg.draw_line(10, maxHeight + 10 - y, 270, maxHeight + 10 - y, gray);
    }
    
    histImg.draw_rectangle(0, 0, 279, maxHeight + 29, white, 1);
    histImg.draw_line(10, maxHeight + 10, 270, maxHeight + 10, black);
    histImg.draw_line(10, 10, 10, maxHeight + 10, black);
    
    int numBins = histogram.size();
    float binWidth = 260.0f / numBins;
    for (int i = 0; i < numBins; i++) {
        int barHeight = static_cast<int>((static_cast<double>(histogram[i]) / maxFreq) * maxHeight);
        if (barHeight > 0) {
            histImg.draw_rectangle(10 + static_cast<int>(i * binWidth), maxHeight + 10 - barHeight, 
                                   10 + static_cast<int>((i + 1) * binWidth), maxHeight + 10, 
                                   black, 1.0f);
        }
    }
    
    histImg.draw_text(5, maxHeight + 20, "0", black);
    histImg.draw_text(125, maxHeight + 20, to_string(numBins / 2).c_str(), black);
    histImg.draw_text(255, maxHeight + 20, to_string(numBins - 1).c_str(), black);
    
    return histImg;
}

void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform (index starting from 0)" << std::endl;
    std::cerr << "  -d : select device (index starting from 0)" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    string image_filename = "mdr16-gs.pgm";
    int selected_platform = -1;
    int selected_device = -1;
    bool list_devices = false;

    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "-h") { print_help(); return 0; }
        if (string(argv[i]) == "-l") { list_devices = true; }
        if (string(argv[i]) == "-p" && i + 1 < argc) { selected_platform = stoi(argv[++i]); }
        if (string(argv[i]) == "-d" && i + 1 < argc) { selected_device = stoi(argv[++i]); }
    }

    if (list_devices) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) { std::cerr << "No OpenCL platforms found" << std::endl; return 1; }
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
        // Load image and determine bit depth
        CImg<unsigned short> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");
        
        int width = image_input.width();
        int height = image_input.height();
        int total_pixels = width * height;
        int bit_depth = (image_input.spectrum() == 1 && image_input.max() > 255) ? 16 : 8;
        int NUM_BINS = (bit_depth == 8) ? 256 : 65536;

        std::cout << "Detected bit depth: " << bit_depth << "-bit" << std::endl;

        // OpenCL setup
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) { std::cerr << "No OpenCL platforms found" << std::endl; return 1; }
        
        cl::Platform selected_platform_obj = (selected_platform >= 0 && selected_platform < platforms.size()) 
            ? platforms[selected_platform] : platforms[0];
        std::cout << "Selected platform: " << selected_platform_obj.getInfo<CL_PLATFORM_NAME>() << std::endl;

        cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(selected_platform_obj)(), 0 };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) { std::cerr << "No OpenCL devices found" << std::endl; return 1; }
        
        cl::Device selected_device_obj = (selected_device >= 0 && selected_device < devices.size()) 
            ? devices[selected_device] : devices[0];
        std::cout << "Selected device: " << selected_device_obj.getInfo<CL_DEVICE_NAME>() << std::endl;

        cl::CommandQueue queue(context, selected_device_obj, CL_QUEUE_PROFILING_ENABLE);

        // Load appropriate kernel based on bit depth
        string kernelSource = loadKernelSource(bit_depth == 8 ? "kernels/8_bit.cl" : "kernels/16_bit.cl");
        cl::Program program(context, kernelSource);
        program.build(devices);

        if (bit_depth == 8) {
            std::vector<unsigned char> h_inputImage(total_pixels);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    h_inputImage[y * width + x] = static_cast<unsigned char>(image_input(x, y, 0));
                }
            }
            
            cl::Buffer d_inputImage(context, CL_MEM_READ_ONLY, total_pixels * sizeof(unsigned char));
            cl::Buffer d_outputImage(context, CL_MEM_WRITE_ONLY, total_pixels * sizeof(unsigned char));
            cl::Buffer d_histogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
            cl::Buffer d_cumulativeHistogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
            cl::Buffer d_lut(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));

            std::vector<int> h_histogram(NUM_BINS, 0);
            std::vector<int> h_cumulativeHistogram(NUM_BINS, 0);
            std::vector<int> h_lut(NUM_BINS, 0);
            std::vector<unsigned char> h_outputImage(total_pixels);

            queue.enqueueWriteBuffer(d_inputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned char), h_inputImage.data());
            queue.enqueueFillBuffer(d_histogram, 0, 0, NUM_BINS * sizeof(int));

            cl::Kernel histogramKernel(program, "calculateHistogram");
            cl::Kernel scanKernel(program, "prefixSum");
            cl::Kernel lutKernel(program, "normalizeLUT");
            cl::Kernel applyLUTKernel(program, "applyLUT");

            histogramKernel.setArg(0, d_inputImage);
            histogramKernel.setArg(1, d_histogram);
            histogramKernel.setArg(2, total_pixels);

            size_t workGroupSize = 64;
            size_t globalWorkSize = ((total_pixels + workGroupSize - 1) / workGroupSize) * workGroupSize;

            cl::Event histogramEvent;
            queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(workGroupSize), nullptr, &histogramEvent);
            histogramEvent.wait();

            queue.enqueueReadBuffer(d_histogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_histogram.data());

            scanKernel.setArg(0, d_histogram);
            scanKernel.setArg(1, d_cumulativeHistogram);
            scanKernel.setArg(2, NUM_BINS);

            cl::Event scanEvent;
            queue.enqueueNDRangeKernel(scanKernel, cl::NullRange, cl::NDRange(NUM_BINS), cl::NDRange(std::min(256, NUM_BINS)), nullptr, &scanEvent);
            scanEvent.wait();

            queue.enqueueReadBuffer(d_cumulativeHistogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_cumulativeHistogram.data());

            lutKernel.setArg(0, d_cumulativeHistogram);
            lutKernel.setArg(1, d_lut);
            lutKernel.setArg(2, total_pixels);
            lutKernel.setArg(3, NUM_BINS);

            cl::Event lutEvent;
            queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(NUM_BINS), cl::NullRange, nullptr, &lutEvent);
            lutEvent.wait();

            queue.enqueueReadBuffer(d_lut, CL_TRUE, 0, NUM_BINS * sizeof(int), h_lut.data());

            applyLUTKernel.setArg(0, d_inputImage);
            applyLUTKernel.setArg(1, d_lut);
            applyLUTKernel.setArg(2, d_outputImage);
            applyLUTKernel.setArg(3, total_pixels);

            cl::Event applyEvent;
            queue.enqueueNDRangeKernel(applyLUTKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(workGroupSize), nullptr, &applyEvent);
            applyEvent.wait();

            queue.enqueueReadBuffer(d_outputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned char), h_outputImage.data());

            CImg<unsigned char> image_output(width, height, 1, 1);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    image_output(x, y, 0) = h_outputImage[y * width + x];
                }
            }

            CImgDisplay disp_output(image_output, "Equalized Image");
            CImg<unsigned char> inputHistImg = createHistogramImage(h_histogram);
            CImgDisplay disp_inputHist(inputHistImg, "Input Histogram");
            CImg<unsigned char> cumHistImg = createHistogramImage(h_cumulativeHistogram);
            CImgDisplay disp_cumHist(cumHistImg, "Cumulative Histogram");
            CImg<unsigned char> lutImg = createHistogramImage(h_lut);
            CImgDisplay disp_lut(lutImg, "LUT");

            while (!disp_input.is_closed() && !disp_output.is_closed() && 
                   !disp_inputHist.is_closed() && !disp_cumHist.is_closed() && !disp_lut.is_closed()) {
                CImgDisplay::wait(disp_input, disp_output, disp_inputHist, disp_cumHist, disp_lut);
            }
        } else { // 16-bit
            std::vector<unsigned short> h_inputImage(total_pixels);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    h_inputImage[y * width + x] = image_input(x, y, 0);
                }
            }
            
            cl::Buffer d_inputImage(context, CL_MEM_READ_ONLY, total_pixels * sizeof(unsigned short));
            cl::Buffer d_outputImage(context, CL_MEM_WRITE_ONLY, total_pixels * sizeof(unsigned short));
            cl::Buffer d_histogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
            cl::Buffer d_cumulativeHistogram(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));
            cl::Buffer d_lut(context, CL_MEM_READ_WRITE, NUM_BINS * sizeof(int));

            std::vector<int> h_histogram(NUM_BINS, 0);
            std::vector<int> h_cumulativeHistogram(NUM_BINS, 0);
            std::vector<int> h_lut(NUM_BINS, 0);
            std::vector<unsigned short> h_outputImage(total_pixels);

            queue.enqueueWriteBuffer(d_inputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned short), h_inputImage.data());
            queue.enqueueFillBuffer(d_histogram, 0, 0, NUM_BINS * sizeof(int));

            cl::Kernel histogramKernel(program, "calculateHistogram16");
            cl::Kernel scanKernel(program, "prefixSum16");
            cl::Kernel lutKernel(program, "normalizeLUT16");
            cl::Kernel applyLUTKernel(program, "applyLUT16");

            histogramKernel.setArg(0, d_inputImage);
            histogramKernel.setArg(1, d_histogram);
            histogramKernel.setArg(2, total_pixels);

            size_t workGroupSize = 64;
            size_t globalWorkSize = ((total_pixels + workGroupSize - 1) / workGroupSize) * workGroupSize;

            cl::Event histogramEvent;
            queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(workGroupSize), nullptr, &histogramEvent);
            histogramEvent.wait();

            queue.enqueueReadBuffer(d_histogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_histogram.data());

            scanKernel.setArg(0, d_histogram);
            scanKernel.setArg(1, d_cumulativeHistogram);
            scanKernel.setArg(2, NUM_BINS);

            cl::Event scanEvent;
            queue.enqueueNDRangeKernel(scanKernel, cl::NullRange, cl::NDRange(NUM_BINS), cl::NDRange(std::min(256, NUM_BINS)), nullptr, &scanEvent);
            scanEvent.wait();

            queue.enqueueReadBuffer(d_cumulativeHistogram, CL_TRUE, 0, NUM_BINS * sizeof(int), h_cumulativeHistogram.data());

            lutKernel.setArg(0, d_cumulativeHistogram);
            lutKernel.setArg(1, d_lut);
            lutKernel.setArg(2, total_pixels);
            lutKernel.setArg(3, NUM_BINS);

            cl::Event lutEvent;
            queue.enqueueNDRangeKernel(lutKernel, cl::NullRange, cl::NDRange(NUM_BINS), cl::NullRange, nullptr, &lutEvent);
            lutEvent.wait();

            queue.enqueueReadBuffer(d_lut, CL_TRUE, 0, NUM_BINS * sizeof(int), h_lut.data());

            applyLUTKernel.setArg(0, d_inputImage);
            applyLUTKernel.setArg(1, d_lut);
            applyLUTKernel.setArg(2, d_outputImage);
            applyLUTKernel.setArg(3, total_pixels);

            cl::Event applyEvent;
            queue.enqueueNDRangeKernel(applyLUTKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(workGroupSize), nullptr, &applyEvent);
            applyEvent.wait();

            queue.enqueueReadBuffer(d_outputImage, CL_TRUE, 0, total_pixels * sizeof(unsigned short), h_outputImage.data());

            CImg<unsigned short> image_output(width, height, 1, 1);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    image_output(x, y, 0) = h_outputImage[y * width + x];
                }
            }

            CImgDisplay disp_output(image_output, "Equalized Image");
            CImg<unsigned char> inputHistImg = createHistogramImage(h_histogram);
            CImgDisplay disp_inputHist(inputHistImg, "Input Histogram");
            CImg<unsigned char> cumHistImg = createHistogramImage(h_cumulativeHistogram);
            CImgDisplay disp_cumHist(cumHistImg, "Cumulative Histogram");
            CImg<unsigned char> lutImg = createHistogramImage(h_lut);
            CImgDisplay disp_lut(lutImg, "LUT");

            while (!disp_input.is_closed() && !disp_output.is_closed() && 
                   !disp_inputHist.is_closed() && !disp_cumHist.is_closed() && !disp_lut.is_closed()) {
                CImgDisplay::wait(disp_input, disp_output, disp_inputHist, disp_cumHist, disp_lut);
            }
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
