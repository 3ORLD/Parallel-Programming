__kernel void calculateHistogram16(__global const unsigned short* image, 
                                 __global int* histogram,
                                 const int totalPixels) {
    int gid = get_global_id(0);
    
    // Process pixels
    if (gid < totalPixels) {
        unsigned short pixelValue = image[gid];
        atomic_inc(&histogram[pixelValue]);
    }
}

// Modified prefix sum for handling large arrays with limited local memory
__kernel void prefixSum16(__global int* input, 
                        __global int* output,
                        const int n) {
    // Use a smaller local buffer size that fits in local memory
    __local int temp[256];
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);
    int global_id = get_global_id(0);
    
    // Process data in chunks
    int chunk_size = 256;
    int num_chunks = (n + chunk_size - 1) / chunk_size;
    
    // Process each chunk
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int offset = chunk * chunk_size;
        int chunk_end = min(offset + chunk_size, n);
        int chunk_elements = chunk_end - offset;
        
        // Load chunk data into local memory
        if (local_id < chunk_elements && offset + local_id < n) {
            temp[local_id] = input[offset + local_id];
        } else if (local_id < chunk_elements) {
            temp[local_id] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform local prefix sum (Hillis-Steele algorithm)
        for (int stride = 1; stride < chunk_elements; stride *= 2) {
            int index = local_id;
            int addIndex = index - stride;
            int val = 0;
            
            if (addIndex >= 0 && index < chunk_elements) {
                val = temp[addIndex];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if (index < chunk_elements) {
                temp[index] += val;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Write results back to global memory
        if (local_id < chunk_elements && offset + local_id < n) {
            output[offset + local_id] = temp[local_id];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    // Combine chunk results - perform a second pass to add offsets
    if (global_id == 0) {
        int running_sum = 0;
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int offset = chunk * chunk_size;
            int chunk_end = min(offset + chunk_size, n);
            int last_index = chunk_end - 1;
            
            int chunk_sum = output[last_index];
            output[last_index] = running_sum;
            running_sum += chunk_sum;
            
            for (int i = last_index - 1; i >= offset; i--) {
                int temp = output[i];
                output[i] = running_sum - (chunk_sum - output[i]);
            }
        }
    }
}

__kernel void normalizeLUT16(__global int* cumulativeHistogram, 
                           __global int* lut,
                           const int totalPixels,
                           const int numBins) {
    int gid = get_global_id(0);
    if (gid < numBins) {
        // Ensure we don't divide by zero
        float scale = (totalPixels > 0) ? ((float)cumulativeHistogram[gid] / totalPixels * 65535.0f) : 0.0f;
        lut[gid] = (int)(scale);
    }
}

__kernel void applyLUT16(__global const unsigned short* inputImage, 
                        __global const int* lut,
                        __global unsigned short* outputImage,
                        const int totalPixels) {
    int gid = get_global_id(0);
    if (gid < totalPixels) {
        unsigned short pixelValue = inputImage[gid];
        outputImage[gid] = (unsigned short)lut[pixelValue];
    }
}
