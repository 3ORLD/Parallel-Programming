#include <iostream>
#include <vector>
#include <chrono>
#include "Utils.h" // Assumed to include OpenCL headers
#include "CImg.h"

using namespace cimg_library;
using namespace std;

// Loads OpenCL kernel source code from a file
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

// Creates a histogram visualization image
CImg<unsigned char> createHistogramImage(const vector<int>& histogram, int maxHeight = 200) {
    int maxFreq = *max_element(histogram.begin(), histogram.end());
    CImg<unsigned char> histImg(280, maxHeight + 30, 1, 3, 255);
    const unsigned char black[] = {0, 0, 0}, white[] = {255, 255, 255}, gray[] = {169, 169, 169};

    histImg.draw_rectangle(10, maxHeight + 10, 270, maxHeight + 30, black, 1.0f); // Base rectangle
    histImg.draw_rectangle(10, 10, 270, maxHeight + 10, white, 1.0f); // Background
    for (int y = 0; y < maxHeight; y += 20) {
        histImg.draw_line(10, maxHeight + 10 - y, 270, maxHeight + 10 - y, gray); // Grid lines
    }
    histImg.draw_rectangle(0, 0, 279, maxHeight + 29, white, 1); // Border
    histImg.draw_line(10, maxHeight + 10, 270, maxHeight + 10, black); // X-axis
    histImg.draw_line(10, 10, 10, maxHeight + 10, black); // Y-axis

    int numBins = histogram.size();
    float binWidth = 260.0f / numBins;
    for (int i = 0; i < numBins; i++) {
        int barHeight = (maxFreq > 0) ? static_cast<int>((static_cast<double>(histogram[i]) / maxFreq) * maxHeight) : 0;
        if (barHeight > 0) {
            histImg.draw_rectangle(10 + static_cast<int>(i * binWidth), maxHeight + 10 - barHeight,
                                   10 + static_cast<int>((i + 1) * binWidth), maxHeight + 10, black, 1.0f);
        }
    }
    histImg.draw_text(5, maxHeight + 20, "0", black);
    histImg.draw_text(125, maxHeight + 20, to_string(numBins / 2).c_str(), black);
    histImg.draw_text(255, maxHeight + 20, to_string(numBins - 1).c_str(), black);
    return histImg;
}

// Prints command-line usage instructions
void print_help() {
    cerr << "Usage: -p <platform> -d <device> -l (list devices) -b <bins> -c (color) -hp (high-precision 16-bit) -h (help)" << endl;
}

int main(int argc, char **argv) {
    string image_filename = "mdr16.ppm";
    int selected_platform = 0, selected_device = 0, num_bins = -1;
    bool list_devices = false, use_color = false, high_precision_16bit = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "-h") { print_help(); return 0; }
        if (string(argv[i]) == "-l") { list_devices = true; }
        if (string(argv[i]) == "-p" && i + 1 < argc) { selected_platform = stoi(argv[++i]); }
        if (string(argv[i]) == "-d" && i + 1 < argc) { selected_device = stoi(argv[++i]); }
        if (string(argv[i]) == "-b" && i + 1 < argc) { num_bins = stoi(argv[++i]); }
        if (string(argv[i]) == "-c") { use_color = true; }
        if (string(argv[i]) == "-hp") { high_precision_16bit = true; }
    }

    // List available platforms and devices if requested
    if (list_devices) {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (size_t i = 0; i < platforms.size(); ++i) {
            cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
            vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (size_t j = 0; j < devices.size(); ++j) {
                cout << "  Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << endl;
            }
        }
        return 0;
    }

    try {
        // Load and display input image
        CImg<unsigned short> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        int width = image_input.width(), height = image_input.height();
        int total_pixels = width * height;
        int channels = image_input.spectrum();
        use_color = use_color || (channels > 1);

        cout << "Image has " << channels << " channels" << endl;

        // Determine bit depth and bin settings
        int bit_depth = (image_input.max() > 255) ? 16 : 8;
        int max_value = (bit_depth == 8) ? 255 : 65535;
        int max_bins = (bit_depth == 8) ? 256 : (high_precision_16bit ? 65536 : 256);
        num_bins = (num_bins > 0) ? min(num_bins, max_bins) : max_bins;

        cout << "Bit depth: " << bit_depth << "-bit, Channels: " << channels << ", Bins: " << num_bins << endl;

        // Setup OpenCL platform and device
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[selected_platform];
        cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;

        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[selected_device];
        cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl;

        cl_int err = CL_SUCCESS;
        cl::CommandQueue queue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create command queue: " << err << endl;
            return 1;
        }

        // Load kernel source based on bit depth
        string kernelSource = loadKernelSource(bit_depth == 8 ? "kernels/8_bit.cl" : "kernels/16_bit.cl");
        cl::Program program(context, kernelSource);
        cl_int buildErr = program.build({device});
        if (buildErr != CL_SUCCESS) {
            cerr << "Program build error: " << buildErr << endl;
            cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
            return 1;
        }

        // Data structures for histograms and output
        vector<vector<int>> histograms(channels, vector<int>(num_bins, 0));
        vector<vector<int>> cum_histograms(channels, vector<int>(num_bins, 0));
        vector<vector<int>> hs_cum_histograms(channels, vector<int>(num_bins, 0));
        vector<vector<int>> luts(channels, vector<int>(num_bins, 0));
        CImg<unsigned short> final_output(width, height, 1, channels);

        // Start total execution timer
        auto total_start = chrono::high_resolution_clock::now();

        for (int c = 0; c < channels; c++) {
            cout << "\nProcessing Channel " << c << "..." << endl;
            vector<unsigned short> h_input(total_pixels);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    h_input[y * width + x] = image_input(x, y, 0, c);
                }
            }

            // OpenCL buffers
            cl::Buffer d_input(context, CL_MEM_READ_ONLY, total_pixels * sizeof(unsigned short));
            cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, total_pixels * sizeof(unsigned short));
            cl::Buffer d_hist(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));
            cl::Buffer d_cum_hist(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));
            cl::Buffer d_hs_cum_hist(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));
            cl::Buffer d_lut(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));

            // Memory transfer to device (input)
            auto t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueWriteBuffer(d_input, CL_TRUE, 0, total_pixels * sizeof(unsigned short), h_input.data());
            auto t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Memory Write Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            queue.enqueueFillBuffer(d_hist, 0, 0, num_bins * sizeof(int));

            // Histogram calculation kernel
            cl::Kernel hist_kernel(program, bit_depth == 8 ? "calculateHistogram" : "calculateHistogram16");
            hist_kernel.setArg(0, d_input);
            hist_kernel.setArg(1, d_hist);
            hist_kernel.setArg(2, total_pixels);
            hist_kernel.setArg(3, num_bins);
            hist_kernel.setArg(4, max_value);

            size_t local_size = min(256, num_bins); // Work-group size
            size_t global_size = ((total_pixels + local_size - 1) / local_size) * local_size;

            auto t1 = chrono::high_resolution_clock::now();
            cl_int kernelErr = queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
            if (kernelErr != CL_SUCCESS) {
                cerr << "Histogram kernel enqueue error: " << kernelErr << endl;
                return 1;
            }
            cl_int finishErr = queue.finish();
            if (finishErr != CL_SUCCESS) {
                cerr << "Queue finish error after histogram: " << finishErr << endl;
                return 1;
            }
            auto t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Histogram Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            // Read histogram back to host
            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_hist, CL_TRUE, 0, num_bins * sizeof(int), histograms[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Histogram Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Blelloch Scan
            cl::Kernel scan_kernel(program, bit_depth == 8 ? "prefixSum" : "prefixSum16");
            scan_kernel.setArg(0, d_hist);
            scan_kernel.setArg(1, d_cum_hist);
            scan_kernel.setArg(2, num_bins);

            size_t scan_local_size = min(256, num_bins);
            size_t scan_global_size = ((num_bins + scan_local_size - 1) / scan_local_size) * scan_local_size;

            t1 = chrono::high_resolution_clock::now();
            kernelErr = queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(scan_global_size), cl::NDRange(scan_local_size));
            if (kernelErr != CL_SUCCESS) {
                cerr << "Prefix sum kernel enqueue error: " << kernelErr << endl;
                return 1;
            }
            finishErr = queue.finish();
            if (finishErr != CL_SUCCESS) {
                cerr << "Queue finish error after prefix sum: " << finishErr << endl;
                return 1;
            }
            t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Blelloch Scan Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_cum_hist, CL_TRUE, 0, num_bins * sizeof(int), cum_histograms[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Blelloch Scan Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Hillis-Steele Scan
            cl::Kernel hs_scan_kernel(program, bit_depth == 8 ? "hillisSteeleScan" : "hillisSteeleScan16");
            hs_scan_kernel.setArg(0, d_hist);
            hs_scan_kernel.setArg(1, d_hs_cum_hist);
            hs_scan_kernel.setArg(2, num_bins);

            t1 = chrono::high_resolution_clock::now();
            kernelErr = queue.enqueueNDRangeKernel(hs_scan_kernel, cl::NullRange, cl::NDRange(scan_global_size), cl::NDRange(scan_local_size));
            if (kernelErr != CL_SUCCESS) {
                cerr << "Hillis-Steele kernel enqueue error: " << kernelErr << endl;
                return 1;
            }
            finishErr = queue.finish();
            if (finishErr != CL_SUCCESS) {
                cerr << "Queue finish error after Hillis-Steele: " << finishErr << endl;
                return 1;
            }
            t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Hillis-Steele Scan Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_hs_cum_hist, CL_TRUE, 0, num_bins * sizeof(int), hs_cum_histograms[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Hillis-Steele Scan Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Normalize LUT using Blelloch scan results
            cl::Kernel lut_kernel(program, bit_depth == 8 ? "normalizeLUT" : "normalizeLUT16");
            lut_kernel.setArg(0, d_cum_hist);
            lut_kernel.setArg(1, d_lut);
            lut_kernel.setArg(2, total_pixels);
            lut_kernel.setArg(3, num_bins);
            lut_kernel.setArg(4, max_value);

            kernelErr = queue.enqueueNDRangeKernel(lut_kernel, cl::NullRange, cl::NDRange(num_bins), cl::NullRange);
            if (kernelErr != CL_SUCCESS) {
                cerr << "LUT kernel enqueue error: " << kernelErr << endl;
                return 1;
            }
            finishErr = queue.finish();
            if (finishErr != CL_SUCCESS) {
                cerr << "Queue finish error after LUT creation: " << finishErr << endl;
                return 1;
            }

            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_lut, CL_TRUE, 0, num_bins * sizeof(int), luts[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " LUT Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Apply LUT to equalize image
            cl::Kernel apply_kernel(program, bit_depth == 8 ? "applyLUT" : "applyLUT16");
            apply_kernel.setArg(0, d_input);
            apply_kernel.setArg(1, d_lut);
            apply_kernel.setArg(2, d_output);
            apply_kernel.setArg(3, total_pixels);
            apply_kernel.setArg(4, num_bins);

            t1 = chrono::high_resolution_clock::now();
            kernelErr = queue.enqueueNDRangeKernel(apply_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
            if (kernelErr != CL_SUCCESS) {
                cerr << "Apply LUT kernel enqueue error: " << kernelErr << endl;
                return 1;
            }
            finishErr = queue.finish();
            if (finishErr != CL_SUCCESS) {
                cerr << "Queue finish error after apply LUT: " << finishErr << endl;
                return 1;
            }
            t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Apply LUT Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            // Read equalized image back to host
            vector<unsigned short> h_output(total_pixels);
            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_output, CL_TRUE, 0, total_pixels * sizeof(unsigned short), h_output.data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Output Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    final_output(x, y, 0, c) = h_output[y * width + x];
                }
            }
        }

        // End total execution timer
        auto total_end = chrono::high_resolution_clock::now();
        cout << "\nTotal Program Execution Time: " << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count() << "ms" << endl;

        // Display results
        CImgDisplay disp_output(final_output, "Equalized Image");
        vector<CImgDisplay> hist_displays;

        for (int c = 0; c < channels; c++) {
            string hist_title = "Histogram Channel " + to_string(c);
            string cum_hist_title = "Blelloch Cumulative Histogram Channel " + to_string(c);
            string hs_cum_hist_title = "Hillis-Steele Cumulative Histogram Channel " + to_string(c);
            string lut_title = "LUT Channel " + to_string(c);
            hist_displays.push_back(CImgDisplay(createHistogramImage(histograms[c]), hist_title.c_str()));
            hist_displays.push_back(CImgDisplay(createHistogramImage(cum_histograms[c]), cum_hist_title.c_str()));
            hist_displays.push_back(CImgDisplay(createHistogramImage(hs_cum_histograms[c]), hs_cum_hist_title.c_str()));
            hist_displays.push_back(CImgDisplay(createHistogramImage(luts[c]), lut_title.c_str()));
        }

        while (!disp_input.is_closed() && !disp_output.is_closed()) {
            CImgDisplay::wait_all();
        }

    } catch (const cl::Error& e) {
        cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")" << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
