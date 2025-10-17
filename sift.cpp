#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>

#include "sift.hpp"
#include "image.hpp"



//ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
//                                            int num_octaves, int scales_per_octave)
//{
//    // assume initial sigma is 1.0 (after resizing) and smooth
//    // the image with sigma_diff to reach requried base_sigma
//    float base_sigma = sigma_min / MIN_PIX_DIST;
//    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
//    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
//    base_img = gaussian_blur(base_img, sigma_diff);
//
//    int imgs_per_octave = scales_per_octave + 3;
//
//    // determine sigma values for bluring
//    float k = std::pow(2, 1.0/scales_per_octave);
//    std::vector<float> sigma_vals {base_sigma};
//    for (int i = 1; i < imgs_per_octave; i++) {
//        float sigma_prev = base_sigma * std::pow(k, i-1);
//        float sigma_total = k * sigma_prev;
//        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
//    }
//
//    // create a scale space pyramid of gaussian images
//    // images in each octave are half the size of images in the previous one
//    ScaleSpacePyramid pyramid = {
//        num_octaves,
//        imgs_per_octave,
//        std::vector<std::vector<Image>>(num_octaves)
//    };
//    
//    for (int i = 0; i < num_octaves; i++) {
//        pyramid.octaves[i].reserve(imgs_per_octave);
//        pyramid.octaves[i].push_back(std::move(base_img));
//        for (int j = 1; j < sigma_vals.size(); j++) {
//            const Image& prev_img = pyramid.octaves[i].back();
//            pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
//        }
//        // prepare base image for next octave
//        const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
//        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
//                                        Interpolation::NEAREST);
//    }
//
//    return pyramid;
//}

ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach required base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for blurring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            // Use cached kernel version for performance
            pyramid.octaves[i].push_back(gaussian_blur_with_cached_kernel(prev_img, sigma_vals[j]));
        }
        // prepare base image for next octave
        if (i < num_octaves - 1) {
            const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
            base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2, Interpolation::NEAREST);
            //base_img = downsample_2x(next_base_img);
        }
    }

    return pyramid;
}


// generate pyramid of difference of gaussians (DoG) images
//ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
//{
//    ScaleSpacePyramid dog_pyramid = {
//        img_pyramid.num_octaves,
//        img_pyramid.imgs_per_octave - 1,
//        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
//    };
//    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
//        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
//        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
//            Image diff = img_pyramid.octaves[i][j];
//            #pragma omp parallel for schedule(static)
//            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
//                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
//            }
//            dog_pyramid.octaves[i].push_back(diff);
//        }
//    }
//    return dog_pyramid;
//}
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid) {
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            const Image& curr = img_pyramid.octaves[i][j];
            const Image& prev = img_pyramid.octaves[i][j - 1];
            Image diff(curr.width, curr.height, 1);
            int vec_size = 8;  // AVX2: 8 floats
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < diff.height; row++) {  // Parallel over rows for better cache
                const float* curr_row = &curr.data[row * curr.width];
                const float* prev_row = &prev.data[row * prev.width];
                float* diff_row = &diff.data[row * diff.width];
                int x = 0;
                for (; x <= diff.width - vec_size; x += vec_size) {
                    __m256 curr_vec = _mm256_loadu_ps(curr_row + x);  // Unaligned load
                    __m256 prev_vec = _mm256_loadu_ps(prev_row + x);
                    __m256 diff_vec = _mm256_sub_ps(curr_vec, prev_vec);
                    _mm256_storeu_ps(diff_row + x, diff_vec);
                }
                // Scalar remainder (safe bounds)
                for (; x < diff.width; x++) {
                    diff_row[x] = curr_row[x] - prev_row[x];
                }
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}

ScaleSpacePyramid generate_dog_pyramid_fused(const Image& input_img,
                                             float sigma_min,
                                             int num_octaves,
                                             int scales_per_octave)
{
    // Step 1: Pre-compute ALL sigma values for the entire pyramid
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = input_img.resize(input_img.width*2, input_img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);
    
    int imgs_per_octave = scales_per_octave + 3;
    float k = std::pow(2, 1.0/scales_per_octave);
    
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }
    
    // Step 2: Build DoG pyramid directly (fused blur + subtract)
    ScaleSpacePyramid dog_pyramid = {
        num_octaves,
        imgs_per_octave - 1,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    Image prev_gaussian = std::move(base_img);
    
    for (int octave = 0; octave < num_octaves; octave++) {
        dog_pyramid.octaves[octave].reserve(imgs_per_octave - 1);
        
        for (int scale = 1; scale < imgs_per_octave; scale++) {
            // FUSED OPERATION: Blur + Subtract in single pass
            Image curr_gaussian = gaussian_blur(prev_gaussian, sigma_vals[scale]);
            
            // Compute DoG directly while curr_gaussian is hot in cache
            Image dog(curr_gaussian.width, curr_gaussian.height, 1);
            
            int width = curr_gaussian.width;
            int height = curr_gaussian.height;
            const float* curr_data = curr_gaussian.data;
            const float* prev_data = prev_gaussian.data;
            float* dog_data = dog.data;
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < width * height; i++) {
                dog_data[i] = curr_data[i] - prev_data[i];
            }
            
            dog_pyramid.octaves[octave].push_back(dog);
            prev_gaussian = std::move(curr_gaussian);  // Move semantics: no copy
        }
        
        // Prepare for next octave
        if (octave < num_octaves - 1) {
            prev_gaussian = prev_gaussian.resize(prev_gaussian.width/2, 
                                                 prev_gaussian.height/2,
                                                 Interpolation::NEAREST);
        }
    }
    
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

//std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, 
//                                     int start_octave, int end_octave, 
//                                     float contrast_thresh,
//                                     float edge_thresh)
//{
//    std::vector<Keypoint> keypoints;
//    std::vector<std::vector<Keypoint>> thread_keypoints;
//    
//    #pragma omp parallel
//    {
//        #pragma omp single
//        {
//            thread_keypoints.resize(omp_get_num_threads());
//        }
//        
//        #pragma omp for collapse(2) schedule(dynamic)
//        for (int i = start_octave; i < end_octave; i++) {
//            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
//                const std::vector<Image>& octave = dog_pyramid.octaves[i];
//                const Image& img = octave[j];
//                for (int x = 1; x < img.width-1; x++) {
//                    for (int y = 1; y < img.height-1; y++) {
//                        if (std::abs(img.get_pixel(x, y, 0)) < 0.8 * contrast_thresh) {
//                            continue;
//                        }
//                        if (point_is_extremum(octave, j, x, y)) {
//                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
//                            bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh,
//                                                                          edge_thresh);
//                            if (kp_is_valid) {
//                                thread_keypoints[omp_get_thread_num()].push_back(kp);
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    
//    for (const auto& vec : thread_keypoints) {
//        keypoints.insert(keypoints.end(), vec.begin(), vec.end());
//    }
//    
//    return keypoints;
//}

// NEW: Fast extremum check using direct pointers (avoids get_pixel() calls)
inline bool point_is_extremum_fast(const float* img, const float* prev, const float* next,
                                   int x, int y, int width, float center_val)
{
    bool is_min = true, is_max = true;
    
    // Check 26 neighbors using direct pointer arithmetic
    int row_offset = y * width;
    int prev_row_offset = (y - 1) * width;
    int next_row_offset = (y + 1) * width;
    
    // Previous scale plane (3กั3 neighbors)
    for (int dy = -1; dy <= 1; dy++) {
        int prow = (y + dy) * width;
        for (int dx = -1; dx <= 1; dx++) {
            float neighbor = prev[prow + x + dx];
            if (neighbor > center_val) is_max = false;
            if (neighbor < center_val) is_min = false;
            if (!is_min && !is_max) return false;
        }
    }
    
    // Current scale plane (8 neighbors)
    for (int dy = -1; dy <= 1; dy++) {
        int crow = (y + dy) * width;
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;  // Skip center
            float neighbor = img[crow + x + dx];
            if (neighbor > center_val) is_max = false;
            if (neighbor < center_val) is_min = false;
            if (!is_min && !is_max) return false;
        }
    }
    
    // Next scale plane (3กั3 neighbors)
    for (int dy = -1; dy <= 1; dy++) {
        int nrow = (y + dy) * width;
        for (int dx = -1; dx <= 1; dx++) {
            float neighbor = next[nrow + x + dx];
            if (neighbor > center_val) is_max = false;
            if (neighbor < center_val) is_min = false;
            if (!is_min && !is_max) return false;
        }
    }
    
    return true;
}

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, 
                                     int start_octave, int end_octave, 
                                     float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    std::vector<std::vector<Keypoint>> thread_keypoints;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            thread_keypoints.resize(omp_get_num_threads());
        }
        
        // CHANGED: Y-outer, X-inner (row-major order)
        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = start_octave; i < end_octave; i++) {
            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                const std::vector<Image>& octave = dog_pyramid.octaves[i];
                const Image& img = octave[j];
                const Image& prev = octave[j-1];
                const Image& next = octave[j+1];
                
                int width = img.width;
                int height = img.height;
                const float* img_data = img.data;
                const float* prev_data = prev.data;
                const float* next_data = next.data;
                
                float contrast_check_thresh = 0.8f * contrast_thresh;
                int thread_id = omp_get_thread_num();
                
                // CHANGED: Y-OUTER for sequential memory access
                for (int y = 1; y < height - 1; y++) {
                    int row_offset = y * width;
                    
                    for (int x = 1; x < width - 1; x++) {
                        int idx = row_offset + x;
                        float center_val = img_data[idx];
                        
                        // Quick rejection: contrast check
                        if (std::abs(center_val) < contrast_check_thresh) {
                            continue;
                        }
                        
                        // Check if extremum (only if contrast passes)
                        if (point_is_extremum_fast(img_data, prev_data, next_data,
                                                   x, y, width, center_val)) {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            bool kp_is_valid = refine_or_discard_keypoint(kp, octave, 
                                                                          contrast_thresh, edge_thresh);
                            if (kp_is_valid) {
                                thread_keypoints[thread_id].push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (const auto& vec : thread_keypoints) {
        keypoints.insert(keypoints.end(), vec.begin(), vec.end());
    }
    
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image grad(width, height, 2);
            
            const float* src = pyramid.octaves[i][j].data;
            float* gx_out = grad.data;
            float* gy_out = grad.data + width * height;
            
            const __m256 half = _mm256_set1_ps(0.5f);
            const int vec_size = 8;
            
            // ===== OPTIMIZATION: Use local buffers to avoid cache ping-pong =====
            
            #pragma omp parallel for schedule(static)
            for (int y = 1; y < height - 1; y++) {
                const float* row_m1 = &src[(y - 1) * width];
                const float* row = &src[y * width];
                const float* row_p1 = &src[(y + 1) * width];
                
                // Local buffers (stack allocation, cache-friendly)
                std::vector<float> gx_local(width);
                std::vector<float> gy_local(width);
                
                // Main computation loop - write to local buffers
                int x = 1;
                
                // Vectorized loop
                for (; x < width - 1 - vec_size; x += vec_size) {
                    // GX computation
                    __m256 left = _mm256_loadu_ps(&row[x - 1]);
                    __m256 right = _mm256_loadu_ps(&row[x + 1]);
                    __m256 gx_vec = _mm256_mul_ps(_mm256_sub_ps(right, left), half);
                    _mm256_storeu_ps(&gx_local[x], gx_vec);
                    
                    // GY computation
                    __m256 up = _mm256_loadu_ps(&row_m1[x]);
                    __m256 down = _mm256_loadu_ps(&row_p1[x]);
                    __m256 gy_vec = _mm256_mul_ps(_mm256_sub_ps(down, up), half);
                    _mm256_storeu_ps(&gy_local[x], gy_vec);
                }
                
                // Scalar remainder
                for (; x < width - 1; x++) {
                    gx_local[x] = (row[x + 1] - row[x - 1]) * 0.5f;
                    gy_local[x] = (row_p1[x] - row_m1[x]) * 0.5f;
                }
                
                // Sequential write to global memory (cache-friendly!)
                float* gx_row = &gx_out[y * width];
                float* gy_row = &gy_out[y * width];
                
                for (int x = 1; x < width - 1; x++) {
                    gx_row[x] = gx_local[x];
                    gy_row[x] = gy_local[x];
                }
            }
            
            // Handle boundary rows (y=0 and y=height-1)
            for (int x = 1; x < width - 1; x++) {
                gx_out[0 * width + x] = 0.0f;
                gy_out[0 * width + x] = 0.0f;
                gx_out[(height-1) * width + x] = 0.0f;
                gy_out[(height-1) * width + x] = 0.0f;
            }
            
            grad_pyramid.octaves[i].push_back(grad);
        }
    }
    return grad_pyramid;
}

//ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
//{
//    ScaleSpacePyramid grad_pyramid = {
//        pyramid.num_octaves,
//        pyramid.imgs_per_octave,
//        std::vector<std::vector<Image>>(pyramid.num_octaves)
//    };
//    
//    for (int i = 0; i < pyramid.num_octaves; i++) {
//        grad_pyramid.octaves[i].reserve(grad_pyramid.imgs_per_octave);
//        int width = pyramid.octaves[i][0].width;
//        int height = pyramid.octaves[i][0].height;
//        
//        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
//            Image grad(width, height, 2);
//            
//            const float* src = pyramid.octaves[i][j].data;
//            float* gx_out = grad.data;
//            float* gy_out = grad.data + width * height;
//            
//            const __m256 half = _mm256_set1_ps(0.5f);
//            const int vec_size = 8;
//            
//            // Row-wise processing for cache efficiency
//            #pragma omp parallel for schedule(static)
//            for (int y = 1; y < height - 1; y++) {
//                const float* row_m1 = &src[(y - 1) * width];
//                const float* row = &src[y * width];
//                const float* row_p1 = &src[(y + 1) * width];
//                
//                float* gx_row = &gx_out[y * width];
//                float* gy_row = &gy_out[y * width];
//                
//                // Vectorized main loop (no boundary checks)
//                for (int x = 1; x < width - 1 - vec_size; x += vec_size) {
//                    // GX computation
//                    __m256 left = _mm256_loadu_ps(&row[x - 1]);
//                    __m256 right = _mm256_loadu_ps(&row[x + 1]);
//                    __m256 gx_vec = _mm256_mul_ps(_mm256_sub_ps(right, left), half);
//                    _mm256_storeu_ps(&gx_row[x], gx_vec);
//                    
//                    // GY computation
//                    __m256 up = _mm256_loadu_ps(&row_m1[x]);
//                    __m256 down = _mm256_loadu_ps(&row_p1[x]);
//                    __m256 gy_vec = _mm256_mul_ps(_mm256_sub_ps(down, up), half);
//                    _mm256_storeu_ps(&gy_row[x], gy_vec);
//                }
//                
//                // Scalar edges (only 7 pixels per row)
//                for (int x = width - 1 - ((width - 2) % vec_size); x < width - 1; x++) {
//                    gx_row[x] = (row[x + 1] - row[x - 1]) * 0.5f;
//                    gy_row[x] = (row_p1[x] - row_m1[x]) * 0.5f;
//                }
//            }
//            
//            // Handle boundary rows (y=0 and y=height-1)
//            for (int x = 1; x < width - 1; x++) {
//                gx_out[0 * width + x] = 0.0f;
//                gy_out[0 * width + x] = 0.0f;
//                gx_out[(height-1) * width + x] = 0.0f;
//                gy_out[(height-1) * width + x] = 0.0f;
//            }
//            
//            grad_pyramid.octaves[i].push_back(grad);
//        }
//    }
//    return grad_pyramid;
//}


// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    
    #pragma omp parallel for schedule(static) reduction(+:hist[0:N_BINS])
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}


void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    // Precompute transcendental function once
    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);
    float patch_sigma = lambda_desc * kp.sigma;
    
    // new add
    int width = img_grad.width;
    const float* gx_data = img_grad.data;                            // Channel 0
    const float* gy_data = img_grad.data + width * img_grad.height;  // Channel 1
    
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        for (int n = y_start; n <= y_end; n++) {
            // find normalized coords w.r.t. kp position and reference orientation
            float x = ((m*pix_dist - kp.x)*cos_t
                      +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
            float y = (-(m*pix_dist - kp.x)*sin_t
                       +(n*pix_dist - kp.y)*cos_t) / kp.sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                continue;

            // float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float gx = gx_data[n * width + m];
            float gy = gy_data[n * width + m];
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                    /(2*patch_sigma*patch_sigma));
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}


// ---------------------------------------------------

struct OctaveAssignment {
    int start_octave;
    int end_octave;
    long long total_work;
};

struct LoadBalanceAssignment {
    std::vector<int> start_octaves;  // start_octaves[rank] = starting octave
    std::vector<int> end_octaves;    // end_octaves[rank] = ending octave (exclusive)
};

// Compute work units (pixels) for a specific octave
long long compute_octave_work(int octave_idx, int base_width, int base_height, 
                              int scales_per_octave)
{
    // Each octave is 1/4 the size of previous (half width, half height)
    long long width = base_width >> octave_idx;   // Equivalent to / 2^octave_idx
    long long height = base_height >> octave_idx;
    return width * height * scales_per_octave;
}

//LoadBalanceAssignment compute_load_balance(int world_size, 
//                                           int num_octaves, 
//                                           int img_width, int img_height,
//                                           int scales_per_octave)
//{
//    // Step 1: Calculate work for each octave
//    std::vector<long long> octave_work(num_octaves);
//    long long total_work = 0;
//    
//    for (int i = 0; i < num_octaves; i++) {
//        octave_work[i] = compute_octave_work(i, img_width, img_height, scales_per_octave);
//        total_work += octave_work[i];
//    }
//    
//    // Step 2: Use greedy algorithm to assign octaves to processes
//    std::vector<long long> process_work(world_size, 0);
//    std::vector<int> octave_to_process(num_octaves);
//    
//    for (int octave = 0; octave < num_octaves; octave++) {
//        // Find process with minimum current work
//        int min_process = 0;
//        long long min_work = process_work[0];
//        
//        for (int p = 1; p < world_size; p++) {
//            if (process_work[p] < min_work) {
//                min_work = process_work[p];
//                min_process = p;
//            }
//        }
//        
//        // Assign this octave to least-loaded process
//        octave_to_process[octave] = min_process;
//        process_work[min_process] += octave_work[octave];
//    }
//    
//    // Step 3: Convert to contiguous ranges for each process
//    LoadBalanceAssignment assignment;
//    assignment.start_octaves.resize(world_size);
//    assignment.end_octaves.resize(world_size);
//    
//    std::vector<int> first_octave(world_size, num_octaves);  // Initialize to "none"
//    std::vector<int> last_octave(world_size, -1);            // Initialize to "none"
//    
//    for (int octave = 0; octave < num_octaves; octave++) {
//        int proc = octave_to_process[octave];
//        if (octave < first_octave[proc]) {
//            first_octave[proc] = octave;
//        }
//        if (octave > last_octave[proc]) {
//            last_octave[proc] = octave;
//        }
//    }
//    
//    for (int p = 0; p < world_size; p++) {
//        if (first_octave[p] == num_octaves) {
//            // This process got no octaves
//            assignment.start_octaves[p] = 0;
//            assignment.end_octaves[p] = 0;
//        } else {
//            assignment.start_octaves[p] = first_octave[p];
//            assignment.end_octaves[p] = last_octave[p] + 1;
//        }
//    }
//    
//    return assignment;
//}

LoadBalanceAssignment compute_load_balance(int world_size, 
                                                     int num_octaves, 
                                                     int img_width, int img_height,
                                                     int scales_per_octave)
{
    // Calculate work for each octave
    std::vector<long long> octave_work(num_octaves);
    long long total_work = 0;
    
    for (int i = 0; i < num_octaves; i++) {
        long long width = img_width >> i;   // Divide by 2^i
        long long height = img_height >> i;
        octave_work[i] = width * height * scales_per_octave;
        total_work += octave_work[i];
    }
    
    // Use greedy assignment to minimize max workload (not average)
    std::vector<long long> process_work(world_size, 0);
    std::vector<std::vector<int>> process_octaves(world_size);
    
    // Assign octaves in decreasing work order
    std::vector<std::pair<long long, int>> work_idx;
    for (int i = 0; i < num_octaves; i++) {
        work_idx.push_back({octave_work[i], i});
    }
    std::sort(work_idx.rbegin(), work_idx.rend());  // Sort descending
    
    for (auto [work, octave] : work_idx) {
        // Find process with minimum current work
        int min_proc = 0;
        for (int p = 1; p < world_size; p++) {
            if (process_work[p] < process_work[min_proc]) {
                min_proc = p;
            }
        }
        
        process_octaves[min_proc].push_back(octave);
        process_work[min_proc] += work;
    }
    
    // Create contiguous ranges
    LoadBalanceAssignment assignment;
    assignment.start_octaves.resize(world_size);
    assignment.end_octaves.resize(world_size);
    
    for (int p = 0; p < world_size; p++) {
        if (process_octaves[p].empty()) {
            assignment.start_octaves[p] = 0;
            assignment.end_octaves[p] = 0;
        } else {
            int min_oct = *std::min_element(process_octaves[p].begin(), process_octaves[p].end());
            int max_oct = *std::max_element(process_octaves[p].begin(), process_octaves[p].end());
            assignment.start_octaves[p] = min_oct;
            assignment.end_octaves[p] = max_oct + 1;
        }
    }
    
    return assignment;
}

OctaveAssignment compute_octave_assignment(int rank, int world_size, 
                                           int num_octaves, 
                                           int img_width, int img_height,
                                           int scales_per_octave)
{
    // Step 1: Calculate work for each octave
    long long total_work = 0;
    std::vector<long long> octave_work(num_octaves);
    
    for (int i = 0; i < num_octaves; i++) {
        octave_work[i] = compute_octave_work(i, img_width, img_height, scales_per_octave);
        total_work += octave_work[i];
    }
    
    // Step 2: Precompute all process assignments
    std::vector<std::pair<int, int>> process_ranges(world_size, {-1, -1});
    long long work_per_process = total_work / world_size;
    long long current_work = 0;
    int current_process = 0;
    int start_octave = 0;
    
    for (int octave = 0; octave < num_octaves; octave++) {
        current_work += octave_work[octave];
        
        // For non-last processes: move to next if work exceeds target AND we're not at the last octave
        // For last process: gets all remaining octaves
        bool move_to_next = false;
        
        if (current_process < world_size - 1) {
            // If we exceed target work, move to next process
            if (current_work > work_per_process && octave < num_octaves - 1) {
                move_to_next = true;
            }
        }
        
        if (move_to_next) {
            process_ranges[current_process] = {start_octave, octave};
            current_process++;
            start_octave = octave;
            current_work = octave_work[octave];
        }
        
        // Last octave: assign remaining octaves to current process
        if (octave == num_octaves - 1) {
            process_ranges[current_process] = {start_octave, octave + 1};
        }
    }
    
    // Step 3: Return assignment for this rank
    OctaveAssignment assignment;
    assignment.start_octave = 0;
    assignment.end_octave = 0;
    assignment.total_work = 0;
    
    if (rank < world_size && process_ranges[rank].first != -1) {
        assignment.start_octave = process_ranges[rank].first;
        assignment.end_octave = process_ranges[rank].second;
        
        // Calculate total work for this process
        for (int i = assignment.start_octave; i < assignment.end_octave; i++) {
            assignment.total_work += octave_work[i];
        }
    }
    
    return assignment;
}

OctaveAssignment compute_octave_assignment_greedy(int rank, int world_size, 
                                                   int num_octaves, 
                                                   int img_width, int img_height,
                                                   int scales_per_octave)
{
    // Step 1: Calculate work for each octave
    std::vector<long long> octave_work(num_octaves);
    long long total_work = 0;
    
    for (int i = 0; i < num_octaves; i++) {
        octave_work[i] = compute_octave_work(i, img_width, img_height, scales_per_octave);
        total_work += octave_work[i];
    }
    
    // Step 2: Assign each octave to the process with least work (greedy)
    std::vector<long long> process_work(world_size, 0);
    std::vector<int> octave_to_process(num_octaves);
    
    for (int octave = 0; octave < num_octaves; octave++) {
        // Find process with minimum current work
        int min_process = 0;
        long long min_work = process_work[0];
        
        for (int p = 1; p < world_size; p++) {
            if (process_work[p] < min_work) {
                min_work = process_work[p];
                min_process = p;
            }
        }
        
        // Assign this octave to least-loaded process
        octave_to_process[octave] = min_process;
        process_work[min_process] += octave_work[octave];
    }
    
    // Step 3: Find start and end for this rank
    OctaveAssignment assignment;
    assignment.start_octave = num_octaves;  // Default: no octaves
    assignment.end_octave = num_octaves;
    assignment.total_work = 0;
    
    for (int octave = 0; octave < num_octaves; octave++) {
        if (octave_to_process[octave] == rank) {
            if (octave < assignment.start_octave) {
                assignment.start_octave = octave;
            }
            if (octave >= assignment.end_octave) {
                assignment.end_octave = octave + 1;
            }
            assignment.total_work += octave_work[octave];
        }
    }
    
    return assignment;
}

// -----------------------------------------------

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img,
                                                     int rank,
                                                     int size,
                                                     float sigma_min,
                                                     int num_octaves, int scales_per_octave,
                                                     float contrast_thresh, float edge_thresh,
                                                     float lambda_ori, float lambda_desc)
{
    
    assert(img.channels == 1 || img.channels == 3);

    // --- Timing Setup ---
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_step = start_total;
    auto end_step = start_total;

    // --- Grayscale Conversion ---
    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] Grayscale conversion: " << duration.count() << " ms\n";
    }

    // --- Gaussian Pyramid ---
    start_step = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves, scales_per_octave);
    MPI_Barrier(MPI_COMM_WORLD);
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] Gaussian pyramid generation: " << duration.count() << " ms\n";
    }

    // --- Difference of Gaussians (DoG) Pyramid ---
    start_step = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    // NEW SIGNATURE (fused operation)
    //ScaleSpacePyramid dog_pyramid = generate_dog_pyramid_fused(input, sigma_min, num_octaves, scales_per_octave);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] DoG pyramid generation: " << duration.count() << " ms\n";
    }

    // --- Gradient Pyramid ---
    start_step = std::chrono::high_resolution_clock::now();
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    MPI_Barrier(MPI_COMM_WORLD);
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] Gradient pyramid generation: " << duration.count() << " ms\n";
    }

    // --- Keypoint Detection ---
    LoadBalanceAssignment load_assignment;
    
    if (rank == 0) {
        load_assignment = compute_load_balance(size, num_octaves, 
                                               input.width, input.height, 
                                               scales_per_octave);
        
        // Debug output
        std::cout << "[DEBUG] Load balancing info:\n";
        for (int r = 0; r < size; r++) {
            std::cout << "  Process " << r << ": octaves [" 
                      << load_assignment.start_octaves[r] << ", " 
                      << load_assignment.end_octaves[r] << ")\n";
        }
    }
    
    // Broadcast start_octaves array
    std::vector<int> start_octaves_bcast(size);
    std::vector<int> end_octaves_bcast(size);
    
    if (rank == 0) {
        start_octaves_bcast = load_assignment.start_octaves;
        end_octaves_bcast = load_assignment.end_octaves;
    }
    
    MPI_Bcast(start_octaves_bcast.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(end_octaves_bcast.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    int start_octave = start_octaves_bcast[rank];
    int end_octave = end_octaves_bcast[rank];
        
    start_step = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> local_tmp_kps = find_keypoints(dog_pyramid, start_octave, end_octave, contrast_thresh, edge_thresh);
    MPI_Barrier(MPI_COMM_WORLD);
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] Keypoint detection: " << duration.count() << " ms\n";
    }
    
    // -------------------------

    // --- Orientation and Descriptor Generation ---
    start_step = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> local_kps;
    local_kps.reserve(local_tmp_kps.size() * 2);

    std::vector<std::vector<Keypoint>> thread_local_kps;

    #pragma omp parallel
    {
        #pragma omp single
        {
            thread_local_kps.resize(omp_get_num_threads());
        }

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < local_tmp_kps.size(); ++i) {
            Keypoint& kp_tmp = local_tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid, lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp; // Create a new keypoint for each orientation
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                thread_local_kps[omp_get_thread_num()].push_back(kp);
            }
        }
    }

    for (const auto& vec : thread_local_kps) {
        local_kps.insert(local_kps.end(), vec.begin(), vec.end());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_step = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::chrono::duration<double, std::milli> duration = end_step - start_step;
        std::cout << "  [TIMING] Orientation and descriptor generation: " << duration.count() << " ms\n";
    }

    return local_kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}