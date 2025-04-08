/// OpenCL kernel for image processing

__kernel void process_image(__global const uchar* input, 
                           __global uchar* output,
                           const int width,
                           const int height) {
    // Get the work-item's position
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Boundary check
    if (x >= width || y >= height)
        return;
    
    // Calculate 1D position in the buffer
    int pos = y * width + x;
    
    // Simple image inversion (you can replace this with more complex processing)
    output[pos] = 255 - input[pos];
    
    // Example of more complex processing (uncomment and modify as needed):
    /*
    // Apply a simple blur effect
    int sum = 0;
    int count = 0;
    
    // Process 3x3 neighborhood
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int nx = x + i;
            int ny = y + j;
            
            // Check boundaries
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    
    // Calculate average
    output[pos] = sum / count;
    */
}
