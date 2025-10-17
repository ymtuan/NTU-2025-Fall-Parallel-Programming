#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

//Image Image::resize(int new_w, int new_h, Interpolation method) const
//{
//    Image resized(new_w, new_h, this->channels);
//    float value = 0;
//    for (int x = 0; x < new_w; x++) {
//        for (int y = 0; y < new_h; y++) {
//            for (int c = 0; c < resized.channels; c++) {
//                float old_x = map_coordinate(this->width, new_w, x);
//                float old_y = map_coordinate(this->height, new_h, y);
//                if (method == Interpolation::BILINEAR)
//                    value = bilinear_interpolate(*this, old_x, old_y, c);
//                else if (method == Interpolation::NEAREST)
//                    value = nn_interpolate(*this, old_x, old_y, c);
//                resized.set_pixel(x, y, c, value);
//            }
//        }
//    }
//    return resized;
//}
Image Image::resize(int new_w, int new_h, Interpolation method) const {
    Image resized(new_w, new_h, this->channels);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < new_h; y++) {
        float old_y = map_coordinate(this->height, new_h, y);
        for (int x = 0; x < new_w; x++) {
            float old_x = map_coordinate(this->width, new_w, x);
            for (int c = 0; c < resized.channels; c++) {
                float value;
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}


float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

// separable 2D gaussian blur for 1 channel image
//Image gaussian_blur(const Image& img, float sigma)
//{
//    assert(img.channels == 1);
//    
//    int size = std::ceil(6 * sigma);
//    if (size % 2 == 0)
//        size++;
//    int center = size / 2;
//    
//    // Pre-compute kernel
//    Image kernel(size, 1, 1);
//    float sum = 0;
//    for (int k = -size/2; k <= size/2; k++) {
//        float val = std::exp(-(k*k) / (2*sigma*sigma));
//        kernel.set_pixel(center+k, 0, 0, val);
//        sum += val;
//    }
//    for (int k = 0; k < size; k++)
//        kernel.data[k] /= sum;
//
//    Image tmp(img.width, img.height, 1);
//    Image filtered(img.width, img.height, 1);
//
//    // ===== VERTICAL PASS (with prefetching) =====
//    const int vec_size = 8;
//    const float* img_data = img.data;
//    const float* kern_data = kernel.data;
//    float* tmp_data = tmp.data;
//    int width = img.width;
//    int height = img.height;
//
//    #pragma omp parallel for schedule(static)
//    for (int y = 0; y < height; y++) {
//        float* tmp_row = tmp_data + y * width;
//        
//        int x = 0;
//        
//        // Vectorized main loop with prefetching
//        for (; x <= width - vec_size; x += vec_size) {
//            // PREFETCH: Load future rows into cache BEFORE we need them
//            for (int k = 0; k < size; k++) {
//                int iy = y + (-center + k);
//                if (iy < 0) iy = 0;
//                if (iy >= height) iy = height - 1;
//                
//                // Prefetch the row (load into L3 cache ahead of time)
//                _mm_prefetch((const char*)&img_data[iy * width + x], _MM_HINT_T0);
//            }
//            
//            __m256 sum_vec = _mm256_setzero_ps();
//            
//            // Main computation (now data is in cache!)
//            for (int k = 0; k < size; k++) {
//                int iy = y + (-center + k);
//                if (iy < 0) iy = 0;
//                if (iy >= height) iy = height - 1;
//                
//                __m256 vals = _mm256_loadu_ps(&img_data[iy * width + x]);
//                __m256 kern = _mm256_set1_ps(kern_data[k]);
//                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
//            }
//            
//            _mm256_storeu_ps(tmp_row + x, sum_vec);
//        }
//        
//        // Scalar remainder
//        for (; x < width; x++) {
//            float sum = 0.0f;
//            for (int k = 0; k < size; k++) {
//                int iy = y + (-center + k);
//                if (iy < 0) iy = 0;
//                if (iy >= height) iy = height - 1;
//                sum += img_data[iy * width + x] * kern_data[k];
//            }
//            tmp_row[x] = sum;
//        }
//    }
//
//    // ===== HORIZONTAL PASS (optimized with boundary separation) =====
//    float* filtered_data = filtered.data;
//    float* tmp_data_ptr = tmp.data;
//    
//    #pragma omp parallel for schedule(static)
//    for (int y = 0; y < height; y++) {
//        const float* tmp_row = tmp_data_ptr + y * width;
//        float* filt_row = filtered_data + y * width;
//        
//        // Calculate safe interior region boundaries
//        int interior_start = center;
//        int interior_end = width - center;
//        
//        // ===== FAST PATH: Interior region (NO boundary checks) =====
//        int x = interior_start;
//        
//        // Vectorized interior loop - pure SIMD, no branching
//        for (; x <= interior_end - vec_size; x += vec_size) {
//            __m256 sum_vec = _mm256_setzero_ps();
//            
//            for (int k = 0; k < size; k++) {
//                int ix_base = x + (-center + k);
//                // NO clamping needed - indices are guaranteed valid
//                
//                __m256 vals = _mm256_loadu_ps(&tmp_row[ix_base]);
//                __m256 kern = _mm256_set1_ps(kern_data[k]);
//                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
//            }
//            
//            _mm256_storeu_ps(filt_row + x, sum_vec);
//        }
//        
//        // Scalar remainder of interior
//        for (; x < interior_end; x++) {
//            float sum = 0.0f;
//            for (int k = 0; k < size; k++) {
//                int ix = x + (-center + k);
//                sum += tmp_row[ix] * kern_data[k];
//            }
//            filt_row[x] = sum;
//        }
//        
//        // ===== SLOW PATH: Left boundary (with clamping) =====
//        for (int x = 0; x < interior_start; x++) {
//            float sum = 0.0f;
//            for (int k = 0; k < size; k++) {
//                int ix = x + (-center + k);
//                if (ix < 0) ix = 0;
//                if (ix >= width) ix = width - 1;
//                sum += tmp_row[ix] * kern_data[k];
//            }
//            filt_row[x] = sum;
//        }
//        
//        // ===== SLOW PATH: Right boundary (with clamping) =====
//        for (int x = interior_end; x < width; x++) {
//            float sum = 0.0f;
//            for (int k = 0; k < size; k++) {
//                int ix = x + (-center + k);
//                if (ix < 0) ix = 0;
//                if (ix >= width) ix = width - 1;
//                sum += tmp_row[ix] * kern_data[k];
//            }
//            filt_row[x] = sum;
//        }
//    }
//    
//    return filtered;
//}

Image gaussian_blur(const Image& img, float sigma)
{
    assert(img.channels == 1);
    
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    
    // Pre-compute kernel
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++)
        kernel.data[k] /= sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    // ===== VERTICAL PASS (optimized) =====
    const int BLOCK_SIZE = 64;  // Cache block for better locality
    const int vec_size = 8;
    const float* img_data = img.data;
    const float* kern_data = kernel.data;
    float* tmp_data = tmp.data;
    int width = img.width;
    int height = img.height;

    #pragma omp parallel for schedule(static) collapse(1)
    for (int y = 0; y < height; y++) {
        float* tmp_row = tmp_data + y * width;
        
        // Vectorized inner loop - no boundary checks in critical path
        int x = 0;
        
        // Fast path: process bulk without boundary checks
        for (; x <= width - vec_size; x += vec_size) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            #pragma omp simd
            for (int k = 0; k < size; k++) {
                int iy = y + (-center + k);
                // Clamp OUTSIDE the vectorized loop
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;
                
                __m256 vals = _mm256_loadu_ps(&img_data[iy * width + x]);
                __m256 kern = _mm256_set1_ps(kern_data[k]);
                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
            }
            
            _mm256_storeu_ps(tmp_row + x, sum_vec);
        }
        
        // Scalar remainder (small, amortized cost)
        for (; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int iy = y + (-center + k);
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;
                sum += img_data[iy * width + x] * kern_data[k];
            }
            tmp_row[x] = sum;
        }
    }

    // ===== HORIZONTAL PASS (optimized) =====
    // Key insight: Use strided access to tmp, but process sequentially
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const float* tmp_row = tmp_data + y * width;
        float* filt_row = filtered.data + y * width;
        
        // Vectorized bulk processing
        int x = 0;
        for (; x <= width - vec_size; x += vec_size) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int k = 0; k < size; k++) {
                int ix_base = x + (-center + k);
                
                // Clamp all 8 indices before loading
                int ix[8];
                #pragma omp simd
                for (int v = 0; v < vec_size; v++) {
                    int idx = ix_base + v;
                    ix[v] = (idx < 0) ? 0 : (idx >= width ? width - 1 : idx);
                }
                
                // Gather loads (or use conditional approach)
                __m256 vals = _mm256_setr_ps(
                    tmp_row[ix[0]], tmp_row[ix[1]], tmp_row[ix[2]], tmp_row[ix[3]],
                    tmp_row[ix[4]], tmp_row[ix[5]], tmp_row[ix[6]], tmp_row[ix[7]]
                );
                
                __m256 kern = _mm256_set1_ps(kern_data[k]);
                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
            }
            
            _mm256_storeu_ps(filt_row + x, sum_vec);
        }
        
        // Scalar remainder
        for (; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int ix = x + (-center + k);
                if (ix < 0) ix = 0;
                if (ix >= width) ix = width - 1;
                sum += tmp_row[ix] * kern_data[k];
            }
            filt_row[x] = sum;
        }
    }
    
    return filtered;
}



void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}


// -----------

#include <map>

// Global kernel cache (thread-safe with #pragma omp critical)
static std::map<float, std::vector<float>> kernel_cache;

std::vector<float> get_or_compute_kernel(float sigma)
{
    std::vector<float> cached_kernel;
    bool found = false;
    
    // Check if kernel already cached (reader section)
    #pragma omp critical(kernel_cache_read)
    {
        auto it = kernel_cache.find(sigma);
        if (it != kernel_cache.end()) {
            cached_kernel = it->second;  // Copy cached kernel
            found = true;
        }
    }
    
    if (found) {
        return cached_kernel;
    }
    
    // Compute kernel if not cached
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    
    std::vector<float> kernel(size);
    float sum = 0.0f;
    float sigma_sq_inv = 1.0f / (2.0f * sigma * sigma);
    
    #pragma omp simd reduction(+:sum)
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) * sigma_sq_inv);
        kernel[center + k] = val;
        sum += val;
    }
    
    float inv_sum = 1.0f / sum;
    #pragma omp simd
    for (int k = 0; k < size; k++)
        kernel[k] *= inv_sum;
    
    // Cache it (writer section - critical region)
    #pragma omp critical(kernel_cache_write)
    {
        kernel_cache[sigma] = kernel;
    }
    
    return kernel;
}

Image gaussian_blur_with_cached_kernel(const Image& img, float sigma)
{
    assert(img.channels == 1);
    
    // Get kernel from cache (or compute and cache it)
    std::vector<float> kernel = get_or_compute_kernel(sigma);
    int size = kernel.size();
    int center = size / 2;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    // ===== VERTICAL PASS =====
    const int vec_size = 8;
    const float* img_data = img.data;
    const float* kern_data = kernel.data();
    float* tmp_data = tmp.data;
    int width = img.width;
    int height = img.height;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        float* tmp_row = tmp_data + y * width;
        
        int x = 0;
        
        // Vectorized main loop
        for (; x <= width - vec_size; x += vec_size) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int k = 0; k < size; k++) {
                int iy = y + (-center + k);
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;
                
                __m256 vals = _mm256_loadu_ps(&img_data[iy * width + x]);
                __m256 kern = _mm256_set1_ps(kern_data[k]);
                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
            }
            
            _mm256_storeu_ps(tmp_row + x, sum_vec);
        }
        
        // Scalar remainder
        for (; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int iy = y + (-center + k);
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;
                sum += img_data[iy * width + x] * kern_data[k];
            }
            tmp_row[x] = sum;
        }
    }

    // ===== HORIZONTAL PASS =====
    float* filtered_data = filtered.data;
    float* tmp_data_ptr = tmp.data;
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const float* tmp_row = tmp_data_ptr + y * width;
        float* filt_row = filtered_data + y * width;
        
        int x = 0;
        for (; x <= width - vec_size; x += vec_size) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int k = 0; k < size; k++) {
                int ix_base = x + (-center + k);
                
                // Clamp all 8 indices
                int ix[8];
                for (int v = 0; v < vec_size; v++) {
                    int idx = ix_base + v;
                    ix[v] = (idx < 0) ? 0 : (idx >= width ? width - 1 : idx);
                }
                
                __m256 vals = _mm256_setr_ps(
                    tmp_row[ix[0]], tmp_row[ix[1]], tmp_row[ix[2]], tmp_row[ix[3]],
                    tmp_row[ix[4]], tmp_row[ix[5]], tmp_row[ix[6]], tmp_row[ix[7]]
                );
                
                __m256 kern = _mm256_set1_ps(kern_data[k]);
                sum_vec = _mm256_fmadd_ps(vals, kern, sum_vec);
            }
            
            _mm256_storeu_ps(filt_row + x, sum_vec);
        }
        
        // Scalar remainder
        for (; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                int ix = x + (-center + k);
                if (ix < 0) ix = 0;
                if (ix >= width) ix = width - 1;
                sum += tmp_row[ix] * kern_data[k];
            }
            filt_row[x] = sum;
        }
    }
    
    return filtered;
}

Image downsample_2x(const Image& img)
{
    assert(img.channels == 1);
    
    Image result(img.width / 2, img.height / 2, 1);
    
    const float* src = img.data;
    float* dst = result.data;
    int src_width = img.width;
    int dst_width = result.width;
    int dst_height = result.height;
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_height; y++) {
        const float* src_row0 = &src[2*y * src_width];
        const float* src_row1 = &src[(2*y+1) * src_width];
        float* dst_row = &dst[y * dst_width];
        
        // Vectorized downsampling
        int x = 0;
        const int vec_size = 4;  // Process 4 output pixels at a time
        
        for (; x <= dst_width - vec_size; x += vec_size) {
            // Read 8 source pixels (2x4 block), average to 4 output pixels
            __m256 row0_lo = _mm256_loadu_ps(&src_row0[2*x]);
            __m256 row0_hi = _mm256_loadu_ps(&src_row0[2*x + 4]);
            __m256 row1_lo = _mm256_loadu_ps(&src_row1[2*x]);
            __m256 row1_hi = _mm256_loadu_ps(&src_row1[2*x + 4]);
            
            // Interleave and average
            __m256 sum_lo = _mm256_add_ps(row0_lo, row1_lo);
            __m256 sum_hi = _mm256_add_ps(row0_hi, row1_hi);
            
            // Horizontal sum within each pair
            // [a,b,c,d,e,f,g,h] -> [(a+b)/4, (c+d)/4, (e+f)/4, (g+h)/4]
            __m256 quarter = _mm256_set1_ps(0.25f);
            
            __m256 perm_lo = _mm256_permute_ps(sum_lo, _MM_SHUFFLE(2,0,3,1));  // Swap pairs
            __m256 avg_lo = _mm256_mul_ps(_mm256_add_ps(sum_lo, perm_lo), quarter);
            
            __m256 perm_hi = _mm256_permute_ps(sum_hi, _MM_SHUFFLE(2,0,3,1));
            __m256 avg_hi = _mm256_mul_ps(_mm256_add_ps(sum_hi, perm_hi), quarter);
            
            // Blend results
            __m256 result_vec = _mm256_blend_ps(avg_lo, avg_hi, 0xAA);
            _mm256_storeu_ps(&dst_row[x], result_vec);
        }
        
        // Scalar remainder
        for (; x < dst_width; x++) {
            dst_row[x] = (src_row0[2*x] + src_row0[2*x+1] +
                         src_row1[2*x] + src_row1[2*x+1]) * 0.25f;
        }
    }
    
    return result;
}