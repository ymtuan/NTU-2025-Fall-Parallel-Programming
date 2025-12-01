#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>
#include <cuda_runtime.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.1415926535897932384626433832795

#define BLOCK_SIZE 256

// Host types (double precision)
typedef glm::dvec2 vec2;
typedef glm::dvec3 vec3;
typedef glm::dvec4 vec4;
typedef glm::dmat3 mat3;

// Device types (float precision)
typedef glm::vec2 vec2f;
typedef glm::vec3 vec3f;
typedef glm::mat3 mat3f;

// Global device parameters - use float
__constant__ int AA_d;
__constant__ float power_d;
__constant__ int md_iter_d;
__constant__ int ray_step_d;
__constant__ int shadow_step_d;
__constant__ float step_limiter_d;
__constant__ float ray_multiplier_d;
__constant__ float bailout_d;
__constant__ float eps_d;
__constant__ float FOV_d;
__constant__ float far_plane_d;
__constant__ int use_pow8_d;

__constant__ vec3f camera_pos_d;
__constant__ vec3f target_pos_d;
__constant__ vec2f iResolution_d;
__constant__ vec3f camera_forward_d;
__constant__ vec3f camera_side_d;
__constant__ vec3f camera_up_d;
__constant__ vec3f light_dir_d;

// Host-side parameters
int AA = 3;
double power = 8.0;
int md_iter = 24;
int ray_step = 10000;
int shadow_step = 1500;
double step_limiter = 0.2;
double ray_multiplier = 0.1;
double bailout = 2.0;
double eps = 0.0005;
double FOV = 1.5;
double far_plane = 100.;

vec3 camera_pos;
vec3 target_pos;

unsigned int width;
unsigned int height;
vec2 iResolution;

unsigned char* raw_image;

void write_png(const char* filename) {
    // Fast-path PNG: keep RGBA8, disable auto convert, disable filtering,
    // and use store (no compression). Much faster CPU time than default.
    LodePNGState state;
    lodepng_state_init(&state);

    state.encoder.auto_convert = 0; // don't analyze image; we already have RGBA8
    state.info_raw.colortype = LCT_RGBA;
    state.info_raw.bitdepth  = 8;
    state.info_png.color.colortype = LCT_RGBA;
    state.info_png.color.bitdepth  = 8;

    state.encoder.add_id = 0;                 // skip encoder signature chunk
    state.encoder.filter_strategy = LFS_ZERO; // no per-row filtering
    state.encoder.zlibsettings.btype    = 0;  // store (no deflate)
    state.encoder.zlibsettings.use_lz77 = 0;  // no LZ77

    unsigned char* png = nullptr;
    size_t png_size = 0;
    unsigned error = lodepng_encode(&png, &png_size, raw_image, width, height, &state);
    if (!error) {
        error = lodepng_save_file(png, png_size, filename);
    }

    lodepng_state_cleanup(&state);
    if (png) free(png);

    if (error) {
        printf("png error %u: %s\n", error, lodepng_error_text(error));
    }
}

// Simple CUDA error check helper for cleaner host code
#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t _err = (stmt);                                           \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA error %s at %s:%d: %s\n", #stmt, __FILE__, __LINE__,\
                   cudaGetErrorString(_err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Custom fast math helpers - now using float
__device__ __forceinline__ float fast_length(const vec3f& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ vec3f fast_normalize(const vec3f& v) {
    float len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return vec3f(v.x * len, v.y * len, v.z * len);
}

__device__ __forceinline__ float fast_dot(const vec3f& a, const vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float fast_clamp(float x, float a, float b) {
    return fmaxf(a, fminf(x, b));
}

__device__ __forceinline__ vec3f fast_clamp(const vec3f& v, float a, float b) {
    return vec3f(fast_clamp(v.x, a, b), fast_clamp(v.y, a, b), fast_clamp(v.z, a, b));
}

__device__ __forceinline__ vec3f fast_pow(const vec3f& v, const vec3f& p) {
    return vec3f(powf(v.x, p.x), powf(v.y, p.y), powf(v.z, p.z));
}

__device__ __forceinline__ vec3f fast_cos(const vec3f& v) {
    return vec3f(cosf(v.x), cosf(v.y), cosf(v.z));
}

// Helper: fast r^power and dr multiplier with specialization for power==8
__device__ __forceinline__ void bulb_pow(float r, float& r_pow, float& dr_mul, const bool pow8) {
    if (pow8) {
        float r2 = r * r;
        float r4 = r2 * r2;
        float r8 = r4 * r4;
        r_pow = r8;
        dr_mul = 8.f * (r8 / fmaxf(r, 1e-20f));
    } else {
        r_pow = powf(r, power_d);
        dr_mul = power_d * powf(r, power_d - 1.f);
    }
}

// mandelbulb distance function (DE) - float precision
__device__ __forceinline__ float md(vec3f p, float& trap) {
    vec3f v = p;
    float dr = 1.f;
    float r  = fast_length(v);
    trap = r;

    const bool pow8 = (use_pow8_d != 0);

    #pragma unroll 4
    for (int i = 0; i < md_iter_d; ++i) {
        if (r > bailout_d) break;

        float theta = atan2f(v.y, v.x);
        float phi   = asinf(fmaxf(fminf(v.z / fmaxf(r, 1e-20f), 1.f), -1.f));

        float r_pow, dr_mul;
        bulb_pow(r, r_pow, dr_mul, pow8);
        dr = __fmaf_rn(dr_mul, dr, 1.f);

        float s_t, c_t; sincosf(theta * power_d, &s_t, &c_t);
        float s_p, c_p; sincosf(phi   * power_d, &s_p, &c_p);

        v = p + r_pow * vec3f(c_t * c_p, s_t * c_p, -s_p);

        trap = fminf(trap, r);
        r = fast_length(v);
    }
    return 0.5f * logf(r) * r / dr;
}

// scene mapping - rotate 90° around X by swizzle to avoid mat3 multiply
__device__ __forceinline__ float map(vec3f p, float& trap) {
    vec3f rp = vec3f(p.x, -p.z, p.y);
    return md(rp, trap);
}

__device__ __forceinline__ float map(vec3f p) {
    float dmy;
    return map(p, dmy);
}

// Conservative trace - keep original algorithm for accuracy
__device__ float trace(vec3f ro, vec3f rd, float& trap) {
    float t = 0.f;
    for (int i = 0; i < ray_step_d; ++i) {
        float len = map(ro + rd * t, trap);
        if (fabsf(len) < eps_d || t > far_plane_d) break;
        t += len * ray_multiplier_d;
    }
    return t < far_plane_d ? t : -1.f;
}

// Use 6-sample central differences for accuracy
__device__ __forceinline__ vec3f calcNor(vec3f p) {
    const vec2f e = vec2f(eps_d, 0.f);
    vec3f grad = vec3f(
        map(p + e.xyy()) - map(p - e.xyy()),
        map(p + e.yxy()) - map(p - e.yxy()),
        map(p + e.yyx()) - map(p - e.yyx())
    );
    return fast_normalize(grad);
}

__device__ float softshadow(vec3f ro, vec3f rd, float k) {
    float res = 1.0f;
    float t = 0.0f;
    
    #pragma unroll 4
    for (int i = 0; i < shadow_step_d; ++i) {
        float h = map(ro + rd * t);
        res = fminf(res, k * h / fmaxf(t, 1e-6f));
        
        // Only very conservative early exits
        if (res < 0.02f) return 0.02f;
        
        // Only exit if clearly no shadow after significant iteration
        if (i == 200 && res > 0.95f) {
            return fast_clamp(res, 0.02f, 1.f);
        }
        
        t += fast_clamp(h, .001f, step_limiter_d);
        if (t > far_plane_d) break;
    }
    return fast_clamp(res, 0.02f, 1.f);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 8)
render_kernel(unsigned char* __restrict__ output, int width, int height) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= width * height) return;

    int i = gid / width;
    int j = gid % width;

    vec3f fcol = vec3f(0.0f);

    const float inv_AA = 1.0f / (float)AA_d;
    const float inv_res_y = 1.0f / iResolution_d.y;
    const float inv_samples = 1.0f / (float)(AA_d * AA_d);

    const vec3f sd = light_dir_d;

    const float base_uv_x = (-iResolution_d.x + 2.0f * (float)j) * inv_res_y;
    const float base_uv_y = -(-iResolution_d.y + 2.0f * (float)i) * inv_res_y;
    const float uv_step   = 2.0f * inv_res_y * inv_AA;

    for (int m = 0; m < AA_d; ++m) {
        float uv_x_base = base_uv_x + uv_step * (float)m;
        for (int n = 0; n < AA_d; ++n) {
            float uv_x = uv_x_base;
            float uv_y = base_uv_y - uv_step * (float)n;

            vec3f rd = fast_normalize(uv_x * camera_side_d + uv_y * camera_up_d + FOV_d * camera_forward_d);

            float trap;
            float d = trace(camera_pos_d, rd, trap);

            if (d >= 0.f) {
                vec3f pos = camera_pos_d + rd * d;
                vec3f nr = calcNor(pos);

                vec3f c_t = 2.0f * (float)pi * (vec3f(trap - .4f) + vec3f(.0f, .1f, .2f));
                vec3f col = vec3f(.5f) + vec3f(.5f) * fast_cos(c_t);

                float amb = (0.7f + 0.3f * nr.y) *
                            (0.2f + 0.8f * fast_clamp(0.05f * logf(trap), 0.0f, 1.0f));
                
                float dif = fast_dot(sd, nr);
                float spe = 0.f;
                if (dif > 0.f) {
                    float sdw = softshadow(pos + .001f * nr, sd, 16.f);
                    dif = fast_clamp(dif * sdw, 0.f, 1.f);
                    
                    if (dif > 0.01f) {
                        vec3f hal = fast_normalize(sd - rd);
                        spe = powf(fast_clamp(fast_dot(nr, hal), 0.f, 1.f), 32.f) * dif;
                    }
                } else {
                    dif = 0.f;
                }

                col *= vec3f(0.3f) * (.05f + .95f * amb) + vec3f(1.f, .9f, .717f) * dif * 0.8f;
                col = fast_pow(col, vec3f(.7f, .9f, 1.f));
                col += spe * 0.8f;

                col = fast_pow(col, vec3f(.4545f));
                col = fast_clamp(col, 0.f, 1.f);

                fcol += col;
            }
        }
    }

    fcol *= inv_samples * 255.0f;

    int offset = (i * width + j) * 4;
    uchar4 pixel;
    pixel.x = (unsigned char)fcol.r;
    pixel.y = (unsigned char)fcol.g;
    pixel.z = (unsigned char)fcol.b;
    pixel.w = 255;
    
    *reinterpret_cast<uchar4*>(&output[offset]) = pixel;
}

int main(int argc, char** argv) {
    assert(argc == 10);

    // Parse CLI args and build camera basis on host (double precision)
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);

    iResolution = vec2(width, height);

    // Pre-compute camera basis vectors on host
    vec3 cf = glm::normalize(target_pos - camera_pos);
    vec3 cs = glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));
    vec3 cu = glm::normalize(glm::cross(cs, cf));

    // Convert to float for device
    float power_f = (float)power;
    float step_limiter_f = (float)step_limiter;
    float ray_multiplier_f = (float)ray_multiplier;
    float bailout_f = (float)bailout;
    float eps_f = (float)eps;
    float FOV_f = (float)FOV;
    float far_plane_f = (float)far_plane;
    int use_pow8 = (fabs(power - 8.0) < 1e-6) ? 1 : 0;

    // NEW: precompute light direction on host (same as fast_normalize(camera_pos_d))
    vec3 light_dir = glm::normalize(camera_pos);
    vec3f light_dir_f = vec3f((float)light_dir.x, (float)light_dir.y, (float)light_dir.z);

    vec3f camera_pos_f = vec3f((float)camera_pos.x, (float)camera_pos.y, (float)camera_pos.z);
    vec3f target_pos_f = vec3f((float)target_pos.x, (float)target_pos.y, (float)target_pos.z);
    vec2f iResolution_f = vec2f((float)iResolution.x, (float)iResolution.y);
    vec3f cf_f = vec3f((float)cf.x, (float)cf.y, (float)cf.z);
    vec3f cs_f = vec3f((float)cs.x, (float)cs.y, (float)cs.z);
    vec3f cu_f = vec3f((float)cu.x, (float)cu.y, (float)cu.z);

    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(AA_d, &AA, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(power_d, &power_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(md_iter_d, &md_iter, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(ray_step_d, &ray_step, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(shadow_step_d, &shadow_step, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(step_limiter_d, &step_limiter_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(ray_multiplier_d, &ray_multiplier_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(bailout_d, &bailout_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(eps_d, &eps_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(FOV_d, &FOV_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(far_plane_d, &far_plane_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(camera_pos_d, &camera_pos_f, sizeof(vec3f)));
    CUDA_CHECK(cudaMemcpyToSymbol(target_pos_d, &target_pos_f, sizeof(vec3f)));
    CUDA_CHECK(cudaMemcpyToSymbol(iResolution_d, &iResolution_f, sizeof(vec2f)));
    CUDA_CHECK(cudaMemcpyToSymbol(camera_forward_d, &cf_f, sizeof(vec3f)));
    CUDA_CHECK(cudaMemcpyToSymbol(camera_side_d, &cs_f, sizeof(vec3f)));
    CUDA_CHECK(cudaMemcpyToSymbol(camera_up_d, &cu_f, sizeof(vec3f)));
    CUDA_CHECK(cudaMemcpyToSymbol(use_pow8_d, &use_pow8, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(light_dir_d, &light_dir_f, sizeof(vec3f)));

    // Allocate device memory
    unsigned char* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, width * height * 4 * sizeof(unsigned char)));

    // Allocate pinned host memory for faster D2H copy
    CUDA_CHECK(cudaMallocHost(&raw_image, width * height * 4 * sizeof(unsigned char)));

    int threads_per_block = BLOCK_SIZE;
    int num_pixels = width * height;
    int num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    printf("Rendering with %d blocks and %d threads per block\n", num_blocks, threads_per_block);
    printf("Total pixels: %d\n", num_pixels);

    // Launch kernel
    render_kernel<<<num_blocks, threads_per_block>>>(d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host (pinned) and save PNG
    CUDA_CHECK(cudaMemcpy(raw_image, d_output, width * height * 4 * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    write_png(argv[9]);

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(raw_image));

    return 0;
}
