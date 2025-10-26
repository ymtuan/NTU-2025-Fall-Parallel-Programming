#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.1415926535897932384626433832795

// Host types (double precision)
typedef glm::dvec2 vec2;
typedef glm::dvec3 vec3;
typedef glm::dvec4 vec4;
typedef glm::dmat3 mat3;

// Device types (float precision for speed)
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

__constant__ vec3f camera_pos_d;
__constant__ vec3f target_pos_d;
__constant__ vec2f iResolution_d;
__constant__ vec3f camera_forward_d;
__constant__ vec3f camera_side_d;
__constant__ vec3f camera_up_d;

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
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

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

// mandelbulb distance function (DE) - float precision
__device__ float md(vec3f p, float& trap) {
    vec3f v = p;
    float dr = 1.f;
    float r = fast_length(v);
    trap = r;

    #pragma unroll 4
    for (int i = 0; i < md_iter_d; ++i) {
        float theta = atan2f(v.y, v.x) * power_d;
        float phi = asinf(v.z / r) * power_d;
        dr = __fmaf_rn(power_d * powf(r, power_d - 1.f), dr, 1.f);
        
        float r_pow = powf(r, power_d);
        float cos_phi = cosf(phi);
        v = p + r_pow * vec3f(
            cosf(theta) * cos_phi,
            sinf(theta) * cos_phi,
            -sinf(phi)
        );

        trap = fminf(trap, r);
        r = fast_length(v);
        if (r > bailout_d) break;
    }
    return 0.5f * logf(r) * r / dr;
}

// scene mapping
__device__ float map(vec3f p, float& trap, int& ID) {
    const float pi_2 = pi / 2.f;
    vec2f rt = vec2f(cosf(pi_2), sinf(pi_2));
    vec3f rp = mat3f(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) * p;
    ID = 1;
    return md(rp, trap);
}

__device__ float map(vec3f p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// second march: cast shadow
__device__ float softshadow(vec3f ro, vec3f rd, float k) {
    float res = 1.0f;
    float t = 0.f;
    #pragma unroll 8
    for (int i = 0; i < shadow_step_d; ++i) {
        float h = map(ro + rd * t);
        res = fminf(res, k * h / t);
        if (res < 0.02f) return 0.02f;
        t += fast_clamp(h, .001f, step_limiter_d);
    }
    return fast_clamp(res, .02f, 1.f);
}

// Remove __noinline__, let compiler inline for better optimization
__device__ float trace(vec3f ro, vec3f rd, float& trap, int& ID) {
    float t = 0.f;
    
    for (int i = 0; i < ray_step_d; ++i) {
        float len = map(ro + rd * t, trap, ID);
        if (fabsf(len) < eps_d || t > far_plane_d) break;
        t += len * ray_multiplier_d;
    }
    return t < far_plane_d ? t : -1.f;
}

// Higher occupancy kernel - inline everything
__global__ void __launch_bounds__(256, 4) 
render_kernel(unsigned char* output, int width, int height) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= width * height) return;

    int i = gid / width;
    int j = gid % width;

    float fcol_r = 0.0f, fcol_g = 0.0f, fcol_b = 0.0f;

    const float inv_AA = 1.0f / (float)AA_d;
    const float inv_res_y = 1.0f / iResolution_d.y;
    const float inv_samples = 1.0f / (float)(AA_d * AA_d);
    
    for (int m = 0; m < AA_d; ++m) {
        for (int n = 0; n < AA_d; ++n) {
            float px = (float)j + (float)m * inv_AA;
            float py = (float)i + (float)n * inv_AA;

            float uv_x = (-iResolution_d.x + 2.0f * px) * inv_res_y;
            float uv_y = -(-iResolution_d.y + 2.0f * py) * inv_res_y;

            vec3f rd = fast_normalize(uv_x * camera_side_d + uv_y * camera_up_d + FOV_d * camera_forward_d);

            float trap;
            int objID;
            float d = trace(camera_pos_d, rd, trap, objID);

            // Inline shading to avoid spills
            if (d >= 0.f) {
                vec3f pos = camera_pos_d + rd * d;
                
                // Inline calcNor
                vec2f e = vec2f(eps_d, 0.f);
                vec3f nr = fast_normalize(vec3f(
                    map(pos + e.xyy()) - map(pos - e.xyy()),
                    map(pos + e.yxy()) - map(pos - e.yxy()),
                    map(pos + e.yyx()) - map(pos - e.yyx())
                ));
                
                vec3f sd = fast_normalize(camera_pos_d);
                vec3f hal = fast_normalize(sd - rd);

                // Palette
                vec3f c_t = 2.0f * (float)pi * (vec3f(1.f) * (trap - .4f) + vec3f(.0f, .1f, .2f));
                vec3f col = vec3f(.5f) + vec3f(.5f) * fast_cos(c_t);
                
                float amb = (0.7f + 0.3f * nr.y) * 
                            (0.2f + 0.8f * fast_clamp(0.05f * logf(trap), 0.0f, 1.0f));
                float sdw = softshadow(pos + .001f * nr, sd, 16.f);
                float dif = fast_clamp(fast_dot(sd, nr), 0.f, 1.f) * sdw;
                float spe = powf(fast_clamp(fast_dot(nr, hal), 0.f, 1.f), 32.f) * dif;

                // Lighting
                col *= vec3f(0.3f) * (.05f + .95f * amb) + vec3f(1.f, .9f, .717f) * dif * 0.8f;
                col = fast_pow(col, vec3f(.7f, .9f, 1.f));
                col += spe * 0.8f;
                
                // Gamma correction
                col = fast_pow(col, vec3f(.4545f));
                col = fast_clamp(col, 0.f, 1.f);
                
                fcol_r += col.r;
                fcol_g += col.g;
                fcol_b += col.b;
            }
        }
    }

    fcol_r *= inv_samples * 255.0f;
    fcol_g *= inv_samples * 255.0f;
    fcol_b *= inv_samples * 255.0f;

    int offset = (i * width + j) * 4;
    output[offset + 0] = (unsigned char)fcol_r;
    output[offset + 1] = (unsigned char)fcol_g;
    output[offset + 2] = (unsigned char)fcol_b;
    output[offset + 3] = 255;
}

int main(int argc, char** argv) {
    assert(argc == 10);

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
    
    vec3f camera_pos_f = vec3f((float)camera_pos.x, (float)camera_pos.y, (float)camera_pos.z);
    vec3f target_pos_f = vec3f((float)target_pos.x, (float)target_pos.y, (float)target_pos.z);
    vec2f iResolution_f = vec2f((float)iResolution.x, (float)iResolution.y);
    vec3f cf_f = vec3f((float)cf.x, (float)cf.y, (float)cf.z);
    vec3f cs_f = vec3f((float)cs.x, (float)cs.y, (float)cs.z);
    vec3f cu_f = vec3f((float)cu.x, (float)cu.y, (float)cu.z);

    // Copy constants to device
    cudaMemcpyToSymbol(AA_d, &AA, sizeof(int));
    cudaMemcpyToSymbol(power_d, &power_f, sizeof(float));
    cudaMemcpyToSymbol(md_iter_d, &md_iter, sizeof(int));
    cudaMemcpyToSymbol(ray_step_d, &ray_step, sizeof(int));
    cudaMemcpyToSymbol(shadow_step_d, &shadow_step, sizeof(int));
    cudaMemcpyToSymbol(step_limiter_d, &step_limiter_f, sizeof(float));
    cudaMemcpyToSymbol(ray_multiplier_d, &ray_multiplier_f, sizeof(float));
    cudaMemcpyToSymbol(bailout_d, &bailout_f, sizeof(float));
    cudaMemcpyToSymbol(eps_d, &eps_f, sizeof(float));
    cudaMemcpyToSymbol(FOV_d, &FOV_f, sizeof(float));
    cudaMemcpyToSymbol(far_plane_d, &far_plane_f, sizeof(float));
    cudaMemcpyToSymbol(camera_pos_d, &camera_pos_f, sizeof(vec3f));
    cudaMemcpyToSymbol(target_pos_d, &target_pos_f, sizeof(vec3f));
    cudaMemcpyToSymbol(iResolution_d, &iResolution_f, sizeof(vec2f));
    cudaMemcpyToSymbol(camera_forward_d, &cf_f, sizeof(vec3f));
    cudaMemcpyToSymbol(camera_side_d, &cs_f, sizeof(vec3f));
    cudaMemcpyToSymbol(camera_up_d, &cu_f, sizeof(vec3f));

    // Allocate device memory
    unsigned char* d_output;
    cudaMalloc(&d_output, width * height * 4 * sizeof(unsigned char));

    // Allocate host memory
    raw_image = new unsigned char[width * height * 4];

    // Back to 256 for better occupancy
    int threads_per_block = 256;
    int num_pixels = width * height;
    int num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    printf("Rendering with %d blocks and %d threads per block\n", num_blocks, threads_per_block);
    printf("Total pixels: %d\n", num_pixels);

    // Launch kernel
    render_kernel<<<num_blocks, threads_per_block>>>(d_output, width, height);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(raw_image, d_output, width * height * 4 * sizeof(unsigned char), 
               cudaMemcpyDeviceToHost);

    // Save image
    write_png(argv[9]);

    // Cleanup
    cudaFree(d_output);
    delete[] raw_image;

    return 0;
}