#include <iostream>
#include <vector>
#include "CImg.h"
using namespace cimg_library;
using namespace std;

int main(int argc, char **argv) {
    string image_filename = "test.pgm"; // Ensure this image exists in the same folder
    
    try {
        CImg<unsigned char> image_input(image_filename.c_str());  // Loading the image
        CImgDisplay disp_input(image_input, "Input Image");       // Display the image

        // Calculate histogram - create a vector with 256 bins (for 8-bit grayscale image)
        vector<int> histogram(256, 0);
        
        // Count pixel intensities
        int total_pixels = image_input.width() * image_input.height();
        cimg_forXY(image_input, x, y) {
            unsigned char pixel_value = image_input(x, y, 0, 0);  // Get pixel value at (x,y)
            histogram[pixel_value]++; // Increment the corresponding bin
        }

        // Calculate cumulative histogram
        vector<int> cumulative_hist(256, 0);
        cumulative_hist[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulative_hist[i] = cumulative_hist[i - 1] + histogram[i];
        }

        // Normalize and scale the cumulative histogram to create LUT (0-255)
        vector<int> lut(256, 0);
        vector<int> normalized_lut(256, 0);  // Normalized cumulative LUT
        for (int i = 0; i < 256; i++) {
            // Normalize to [0,1] range and then scale to [0,255]
            lut[i] = (int)((double)cumulative_hist[i] / total_pixels * 255);
            normalized_lut[i] = (int)((double)cumulative_hist[i] / total_pixels * 255); // Normalized LUT
        }

        // Apply the LUT (histogram equalization) to the input image
        CImg<unsigned char> image_output(image_input.width(), image_input.height(), 1, 1, 0);
        cimg_forXY(image_input, x, y) {
            unsigned char pixel_value = image_input(x, y, 0, 0);
            image_output(x, y) = lut[pixel_value];  // Apply LUT to get equalized pixel value
        }

        // Create histogram images
        const int hist_height = 200;
        const int margin_top = 30;
        const int margin_bottom = 50;
        const int margin_left = 60;
        const int total_width = margin_left + 256;
        const int total_height = margin_top + hist_height + margin_bottom;

        // Create the original histogram image
        CImg<unsigned char> hist_image(total_width, total_height, 1, 3, 255);
        const unsigned char black[] = {0, 0, 0};
        const unsigned char red[] = {255, 0, 0};
        const unsigned char gray[] = {200, 200, 200};
        
        // Draw the original histogram
        hist_image.draw_line(margin_left, margin_top, margin_left, margin_top + hist_height, red); // y-axis
        hist_image.draw_line(margin_left, margin_top + hist_height, margin_left + 256, margin_top + hist_height, red); // x-axis

        // Draw the bars of the original histogram
        int max_frequency = *max_element(histogram.begin(), histogram.end());
        for (int i = 0; i < 256; i++) {
            int bar_height = (int)((double)histogram[i] / max_frequency * hist_height);
            hist_image.draw_line(margin_left + i, margin_top + hist_height, margin_left + i, margin_top + hist_height - bar_height, black);
        }

        // Display the original histogram
        CImgDisplay disp_hist(hist_image, "Original Histogram");

        // Create the cumulative histogram image
        CImg<unsigned char> cum_hist_image(total_width, total_height, 1, 3, 255);

        // Draw the cumulative histogram
        cum_hist_image.draw_line(margin_left, margin_top, margin_left, margin_top + hist_height, black); // y-axis
        cum_hist_image.draw_line(margin_left, margin_top + hist_height, margin_left + 256, margin_top + hist_height, black); // x-axis

        // Draw the bars of the cumulative histogram
        for (int i = 0; i < 256; i++) {
            int y_top = margin_top + hist_height - ((double)cumulative_hist[i] / total_pixels * hist_height);
            cum_hist_image.draw_line(margin_left + i, y_top, margin_left + i, margin_top + hist_height, black);
        }

        // Display the cumulative histogram
        CImgDisplay disp_cum_hist(cum_hist_image, "Cumulative Histogram");

        // Create the normalized cumulative LUT image
        CImg<unsigned char> norm_cum_hist_image(total_width, total_height, 1, 3, 255);

        // Draw the normalized cumulative LUT image
        norm_cum_hist_image.draw_line(margin_left, margin_top, margin_left, margin_top + hist_height, black); // y-axis
        norm_cum_hist_image.draw_line(margin_left, margin_top + hist_height, margin_left + 256, margin_top + hist_height, black); // x-axis

        // Draw the bars of the normalized cumulative LUT
        for (int i = 0; i < 256; i++) {
            int y_top = margin_top + hist_height - normalized_lut[i] * hist_height / 255;
            norm_cum_hist_image.draw_line(margin_left + i, y_top, margin_left + i, margin_top + hist_height, black);
        }

        // Display the normalized cumulative LUT
        CImgDisplay disp_norm_cum_hist(norm_cum_hist_image, "Normalized Cumulative LUT");

        // Display the equalized image
        CImgDisplay disp_output(image_output, "Equalized Image");

        // Wait for the user to close the displays
        while (!disp_input.is_closed() && !disp_hist.is_closed() && !disp_cum_hist.is_closed() && !disp_norm_cum_hist.is_closed() && !disp_output.is_closed()) {
            disp_input.wait(1);
            disp_hist.wait(1);
            disp_cum_hist.wait(1);
            disp_norm_cum_hist.wait(1);
            disp_output.wait(1);
        }
    }
    catch (const cimg_library::CImgException &e) {
        cerr << "CImg error: " << e.what() << endl;
    }

    return 0;
}


