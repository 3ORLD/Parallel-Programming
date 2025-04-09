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
    cerr << "Usage: -p <platform> -d <device> -t <type: gpu/cpu> -l (list devices) -b <bins> -c (color) -hp (high-precision 16-bit) -h (help) -i <image>" << endl;
}

int main(int argc, char **argv) {
    string image_filename = "mdr16.ppm";
    int selected_platform = 0, selected_device = 0, num_bins = -1;
    bool list_devices = false, use_color = false, high_precision_16bit = false;
    string device_type_str = "gpu"; // Default to GPU

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "-h") { print_help(); return 0; }
        if (string(argv[i]) == "-l") { list_devices = true; }
        if (string(argv[i]) == "-p" && i + 1 < argc) { selected_platform = stoi(argv[++i]); }
        if (string(argv[i]) == "-d" && i + 1 < argc) { selected_device = stoi(argv[++i]); }
        if (string(argv[i]) == "-t" && i + 1 < argc) { device_type_str = string(argv[++i]); }
        if (string(argv[i]) == "-b" && i + 1 < argc) { num_bins = stoi(argv[++i]); }
        if (string(argv[i]) == "-c") { use_color = true; }
        if (string(argv[i]) == "-hp") { high_precision_16bit = true; }
        if (string(argv[i]) == "-i" && i + 1 < argc) { image_filename = string(argv[++i]); }
    }

    // List available platforms and devices if requested
    if (list_devices) {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (size_t i = 0; i < platforms.size(); ++i) {
            cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
            vector<cl::Device> devices;
            try {
                platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
                for (size_t j = 0; j < devices.size(); ++j) {
                    string type = (devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
                    cout << "  Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << " (" << type << ")" << endl;
                }
            } catch (const cl::Error& e) {
                cout << "  No devices available: " << e.what() << " (" << e.err() << ")" << endl;
            }
        }
        return 0;
    }

    try {
        // Load input image
        CImg<unsigned short> image_input(image_filename.c_str());

        // Determine bit depth and channels
        int width = image_input.width(), height = image_input.height();
        int total_pixels = width * height;
        int channels = image_input.spectrum();
        use_color = use_color || (channels > 1);

        int bit_depth = (image_input.max() > 255) ? 16 : 8;
        cout << "Image has " << channels << " channels" << endl;
        cout << "Input Image Min: " << image_input.min() << ", Max: " << image_input.max() << endl;

        // Convert input image for display
        CImg<unsigned char> display_input(width, height, 1, channels);
        if (bit_depth == 8) {
            cimg_forXYC(image_input, x, y, c) {
                display_input(x, y, 0, c) = static_cast<unsigned char>(image_input(x, y, 0, c));
            }
        } else {
            display_input = image_input.get_normalize(0, 255);
        }
        CImgDisplay disp_input(display_input, "Input Image");

        int max_value = (bit_depth == 8) ? 255 : 65535;
        int max_bins = (bit_depth == 8) ? 256 : (high_precision_16bit ? 65536 : 256);
        num_bins = (num_bins > 0) ? min(num_bins, max_bins) : max_bins;

        cout << "Bit depth: " << bit_depth << "-bit, Channels: " << channels << ", Bins: " << num_bins << endl;

        // Setup OpenCL platform
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            cerr << "No OpenCL platforms available on this system." << endl;
            return 1;
        }
        if (selected_platform >= static_cast<int>(platforms.size())) {
            cerr << "Invalid platform index: " << selected_platform << ". Only " << platforms.size() << " platforms available." << endl;
            return 1;
        }
        cl::Platform platform = platforms[selected_platform];
        cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;

        // Determine device type from command-line argument
        cl_device_type requested_type = (device_type_str == "cpu") ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

        // Get all devices for the platform first
        vector<cl::Device> devices;
        try {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (const cl::Error& e) {
            cerr << "Failed to retrieve devices on platform " << selected_platform << ": " << e.what() << " (" << e.err() << ")" << endl;
            return 1;
        }

        if (devices.empty()) {
            cerr << "No devices available on platform " << selected_platform << ". Available platforms and devices:" << endl;
            for (size_t i = 0; i < platforms.size(); ++i) {
                cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
                vector<cl::Device> avail_devices;
                try {
                    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &avail_devices);
                    for (size_t j = 0; j < avail_devices.size(); ++j) {
                        string type = (avail_devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
                        cout << "  Device " << j << ": " << avail_devices[j].getInfo<CL_DEVICE_NAME>() << " (" << type << ")" << endl;
                    }
                } catch (const cl::Error& e) {
                    cout << "  No devices available: " << e.what() << " (" << e.err() << ")" << endl;
                }
            }
            return 1;
        }

        // Filter devices by requested type
        vector<cl::Device> filtered_devices;
        for (const auto& dev : devices) {
            if (dev.getInfo<CL_DEVICE_TYPE>() == requested_type) {
                filtered_devices.push_back(dev);
            }
        }

        if (filtered_devices.empty()) {
            cout << "No " << (requested_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") 
                 << " devices found on platform " << selected_platform << ". Falling back to available device." << endl;
            filtered_devices = devices;
        }

        if (selected_device >= static_cast<int>(filtered_devices.size())) {
            cerr << "Invalid device index: " << selected_device << ". Only " << filtered_devices.size() 
                 << " devices available for type " << (requested_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") 
                 << " on platform " << selected_platform << "." << endl;
            cout << "Available devices on platform " << selected_platform << ":" << endl;
            for (size_t j = 0; j < devices.size(); ++j) {
                string type = (devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
                cout << "  Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << " (" << type << ")" << endl;
            }
            return 1;
        }

        cl::Device device = filtered_devices[selected_device];
        cl_device_type device_type = device.getInfo<CL_DEVICE_TYPE>();
        cout << "Device: " << device.getInfo<CL_DEVICE_NAME>()
             << " (" << (device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << ")" << endl;

        // Create OpenCL context and command queue
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        cl::Context context({device}, properties);
        cl::CommandQueue queue(context, device, 0);

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
        CImg<unsigned short> final_output(width, height, 1, channels, 0); // Initialize to 0

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

            // Debug: Check input range and sample values
            cout << "Channel " << c << " Input Min: " << *min_element(h_input.begin(), h_input.end())
                 << ", Max: " << *max_element(h_input.begin(), h_input.end()) << endl;
            cout << "Sample Input Values (Top-Left, Mid, Bottom-Right): " 
                 << h_input[0] << ", " << h_input[total_pixels / 2] << ", " << h_input[total_pixels - 1] << endl;

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

            size_t local_size = min(static_cast<size_t>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()), static_cast<size_t>(256));
            size_t global_size = ((total_pixels + local_size - 1) / local_size) * local_size;

            auto t1 = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
            queue.finish();
            auto t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Histogram Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            // Read histogram back to host
            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_hist, CL_TRUE, 0, num_bins * sizeof(int), histograms[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Histogram Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Debug: Check histogram
            int hist_sum = 0;
            for (int i = 0; i < num_bins; i++) hist_sum += histograms[c][i];
            cout << "Channel " << c << " Histogram Sum: " << hist_sum << " (should match total_pixels: " << total_pixels << ")" << endl;

            // Blelloch Scan
            cl::Kernel scan_kernel(program, bit_depth == 8 ? "prefixSum" : "prefixSum16");
            scan_kernel.setArg(0, d_hist);
            scan_kernel.setArg(1, d_cum_hist);
            scan_kernel.setArg(2, num_bins);

            size_t scan_local_size = min(static_cast<size_t>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()), static_cast<size_t>(num_bins));
            size_t scan_global_size = ((num_bins + scan_local_size - 1) / scan_local_size) * scan_local_size;

            t1 = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(scan_global_size), cl::NDRange(scan_local_size));
            queue.finish();
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
            queue.enqueueNDRangeKernel(hs_scan_kernel, cl::NullRange, cl::NDRange(scan_global_size), cl::NDRange(scan_local_size));
            queue.finish();
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

            t1 = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(lut_kernel, cl::NullRange, cl::NDRange(num_bins), cl::NullRange);
            queue.finish();
            t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " LUT Normalization Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_lut, CL_TRUE, 0, num_bins * sizeof(int), luts[c].data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " LUT Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Debug: Check LUT
            cout << "Channel " << c << " LUT Min: " << *min_element(luts[c].begin(), luts[c].end())
                 << ", Max: " << *max_element(luts[c].begin(), luts[c].end()) << endl;

            // Apply LUT to equalize image
            cl::Kernel apply_kernel(program, bit_depth == 8 ? "applyLUT" : "applyLUT16");
            apply_kernel.setArg(0, d_input);
            apply_kernel.setArg(1, d_lut);
            apply_kernel.setArg(2, d_output);
            apply_kernel.setArg(3, total_pixels);
            apply_kernel.setArg(4, num_bins);

            t1 = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(apply_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
            queue.finish();
            t2 = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Apply LUT Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

            // Read equalized image back to host
            vector<unsigned short> h_output(total_pixels);
            t_mem_start = chrono::high_resolution_clock::now();
            queue.enqueueReadBuffer(d_output, CL_TRUE, 0, total_pixels * sizeof(unsigned short), h_output.data());
            t_mem_end = chrono::high_resolution_clock::now();
            cout << "Channel " << c << " Output Read Time: " << chrono::duration_cast<chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms" << endl;

            // Debug: Check output range and sample values across the image
            cout << "Channel " << c << " Output Min: " << *min_element(h_output.begin(), h_output.end())
                 << ", Max: " << *max_element(h_output.begin(), h_output.end()) << endl;
            cout << "Sample Output Values (Top-Left, Top-Right, Mid, Bottom-Left, Bottom-Right): " 
                 << h_output[0] << ", " << h_output[width - 1] << ", " << h_output[total_pixels / 2] << ", "
                 << h_output[(height - 1) * width] << ", " << h_output[total_pixels - 1] << endl;

            // Populate final_output with detailed checking
            int non_zero_count = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    unsigned short val = h_output[y * width + x];
                    final_output(x, y, 0, c) = val;
                    if (val > 0) non_zero_count++;
                }
            }
            cout << "Channel " << c << " Non-Zero Pixels in final_output: " << non_zero_count << " / " << total_pixels << endl;
        }

        // End total execution timer
        auto total_end = chrono::high_resolution_clock::now();
        cout << "\nTotal Program Execution Time: " << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count() << "ms" << endl;

        // Debug: Check final output range and samples
        cout << "Final Output Min: " << final_output.min() << ", Max: " << final_output.max() << endl;
        cout << "Sample Final Output Values (Top-Left, Top-Right, Mid, Bottom-Left, Bottom-Right): " 
             << final_output(0, 0, 0, 0) << ", " << final_output(width - 1, 0, 0, 0) << ", " 
             << final_output(width / 2, height / 2, 0, 0) << ", " << final_output(0, height - 1, 0, 0) << ", " 
             << final_output(width - 1, height - 1, 0, 0) << endl;

        // Display results
        CImg<unsigned char> display_output(width, height, 1, channels);
        if (bit_depth == 8) {
            cimg_forXYC(final_output, x, y, c) {
                display_output(x, y, 0, c) = static_cast<unsigned char>(final_output(x, y, 0, c));
            }
        } else {
            display_output = final_output.get_normalize(0, 255);
        }
        CImgDisplay disp_output(display_output, "Equalized Image");

        // Debug: Check display output range and sample values
        cout << "Display Output Min: " << (int)display_output.min() << ", Max: " << (int)display_output.max() << endl;
        cout << "Sample Display Output Values (Top-Left, Top-Right, Mid, Bottom-Left, Bottom-Right): " 
             << (int)display_output(0, 0, 0, 0) << ", " << (int)display_output(width - 1, 0, 0, 0) << ", " 
             << (int)display_output(width / 2, height / 2, 0, 0) << ", " << (int)display_output(0, height - 1, 0, 0) << ", " 
             << (int)display_output(width - 1, height - 1, 0, 0) << endl;

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
