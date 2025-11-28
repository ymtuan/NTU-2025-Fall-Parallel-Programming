#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic> // Keep for host-side threading logic
#include <numeric>

#define HIP_CHECK(err) \
    do { \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

namespace param {
const int n_steps = 200000;
const double dt = 60.0;
const double eps = 1e-3;
const double G = 6.674e-11;
const double planet_radius = 1e7;
const double missile_speed = 1e6;
const int checkpoint_interval = 10000;
}

const int TYPE_NORMAL = 0;
const int TYPE_DEVICE = 1;
#define BLOCK_SIZE 256
#define STRIDE BLOCK_SIZE 
#define UNROLL_FACTOR 16 

// ---------------------------------------------------------
// DEVICE HELPERS
// ---------------------------------------------------------
__device__ __forceinline__ void atomicMinDouble(double* addr, double value) {
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= value) break;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// ---------------------------------------------------------
// KERNEL FOR BROADCASTING INITIAL STATE
// ---------------------------------------------------------
__global__ void broadcast_kernel(double* d_arr, int n_bodies, int batch_size) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= n_bodies) return;

    double val = d_arr[tid];
    
    for (int i = 1; i < batch_size; ++i) {
        d_arr[i * n_bodies + tid] = val;
    }
}

// ---------------------------------------------------------
// BATCHED KERNELS (Problem 3) - FUSED KERNEL
// ---------------------------------------------------------
__global__ void compute_and_update_all_batched_kernel(
    double* d_qx, double* d_qy, double* d_qz,
    double* d_vx, double* d_vy, double* d_vz,
    const double* d_base_mass, const int* d_type,
    int n, int step,
    const int* d_targets, 
    int planet_id, 
    int* d_missile_steps)       
{
    int universe_idx = blockIdx.y;
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;
    
    bool is_active = (i < n);
    int offset = universe_idx * n;
    int global_i = offset + i;

    // --- Part 1: Compute Mass m ---
    double current_m = d_base_mass[i]; 
    bool is_device = (d_type[i] == TYPE_DEVICE);

    if (is_active && is_device) {
        bool destroyed = false;
        int target_id = d_targets[universe_idx];

        if (target_id == i) {
            double px = d_qx[offset + planet_id];
            double py = d_qy[offset + planet_id];
            double pz = d_qz[offset + planet_id];

            double dx = d_qx[global_i] - px;
            double dy = d_qy[global_i] - py;
            double dz = d_qz[global_i] - pz;
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            double missile_dist = (double)step * param::dt * param::missile_speed;
            if (missile_dist * missile_dist > dist_sq) {
                destroyed = true;
                current_m = 0.0;
                atomicMin(&d_missile_steps[universe_idx], step);
            }
        }

        if (!destroyed) {
            double t = (double)step * param::dt;
            current_m = current_m + 0.5 * current_m * fabs(sin(t / 6000.0));
        }
    }

    // --- Part 2: Compute Forces and Update State ---
    double my_qx = 0.0, my_qy = 0.0, my_qz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double my_vx = 0.0, my_vy = 0.0, my_vz = 0.0;

    if (is_active) {
        my_qx = d_qx[global_i];
        my_qy = d_qy[global_i];
        my_qz = d_qz[global_i];
        my_vx = d_vx[global_i];
        my_vy = d_vy[global_i];
        my_vz = d_vz[global_i];
    }

    __shared__ double s_data[4 * BLOCK_SIZE]; 

    const double G_eps_sq = param::G;
    const double eps_sq = param::eps * param::eps;

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int idx = tile * BLOCK_SIZE + tid;
        
        // Load data from Global Memory (for all bodies j in the universe)
        if (idx < n) {
            int load_idx = offset + idx;
            s_data[0 * STRIDE + tid] = d_qx[load_idx];
            s_data[1 * STRIDE + tid] = d_qy[load_idx];
            s_data[2 * STRIDE + tid] = d_qz[load_idx];
            
            // Re-calculate mass for body j (source of force)
            double mass_j = d_base_mass[idx];
            if (d_type[idx] == TYPE_DEVICE) {
                if (d_targets[universe_idx] == idx) {
                    // Check if missile hit step has been recorded (i.e., destroyed)
                    if (d_missile_steps[universe_idx] <= step) {
                        mass_j = 0.0;
                    } else {
                        double t = (double)step * param::dt;
                        mass_j = mass_j + 0.5 * mass_j * fabs(sin(t / 6000.0));
                    }
                } else {
                    double t = (double)step * param::dt;
                    mass_j = mass_j + 0.5 * mass_j * fabs(sin(t / 6000.0));
                }
            }
            s_data[3 * STRIDE + tid] = mass_j;

        } else {
            s_data[3 * STRIDE + tid] = 0.0; 
        }
        __syncthreads();

        if (is_active) {
            #pragma unroll UNROLL_FACTOR
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int other_local_idx = tile * BLOCK_SIZE + j;
                if (other_local_idx >= n) break;
                
                double dx = s_data[0 * STRIDE + j] - my_qx;
                double dy = s_data[1 * STRIDE + j] - my_qy;
                double dz = s_data[2 * STRIDE + j] - my_qz;
                double mass_j_shared = s_data[3 * STRIDE + j];

                // Expose FMA for distance squared
                double dx2 = dx * dx;
                double dy2 = dy * dy;
                double dz2 = dz * dz;
                double dist_sq = dx2 + dy2;
                dist_sq += dz2;
                dist_sq += eps_sq;

                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                
                double f = G_eps_sq * mass_j_shared * dist_inv3;
                
                // Expose FMA for accumulation
                ax += f * dx; 
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    if (is_active) {
        // Update velocity and position
        double v_new_x = my_vx + ax * param::dt;
        double v_new_y = my_vy + ay * param::dt;
        double v_new_z = my_vz + az * param::dt;

        d_vx[global_i] = v_new_x;
        d_vy[global_i] = v_new_y;
        d_vz[global_i] = v_new_z;

        d_qx[global_i] = my_qx + v_new_x * param::dt;
        d_qy[global_i] = my_qy + v_new_y * param::dt;
        d_qz[global_i] = my_qz + v_new_z * param::dt;
    }
}


__global__ void check_collision_batched_kernel(
    const double* qx, const double* qy, const double* qz,
    int planet, int asteroid, int step, int n,
    int* planet_hit_steps) 
{
    int universe_idx = blockIdx.y;
    if (threadIdx.x != 0) return;

    int offset = universe_idx * n;
    double dx = qx[offset + planet] - qx[offset + asteroid];
    double dy = qy[offset + planet] - qy[offset + asteroid];
    double dz = qz[offset + planet] - qz[offset + asteroid];
    double dist_sq = dx*dx + dy*dy + dz*dz;

    if (planet_hit_steps[universe_idx] == -2) {
        if (dist_sq < param::planet_radius * param::planet_radius) {
             atomicCAS(&planet_hit_steps[universe_idx], -2, step);
        }
    }
}

// ---------------------------------------------------------
// SINGLE SIMULATION KERNELS (Prob 1 & 2) - FUSED KERNEL
// ---------------------------------------------------------

// FUSED: Combines mass update and force computation. d_is_zero_mass is const bool*
__global__ void compute_and_update_all_single_kernel(
    double* d_qx, double* d_qy, double* d_qz,
    double* d_vx, double* d_vy, double* d_vz,
    const double* d_base_mass, const int* d_type,
    int n, int step,
    const bool* d_is_zero_mass) // FIX: Changed to const bool*
{
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;
    
    bool is_active = (i < n);
    bool is_p1 = *d_is_zero_mass; // FIX: Direct dereference of const bool*

    // --- Part 1: Compute Mass m ---
    double current_m = d_base_mass[i]; 
    bool is_device = (d_type[i] == TYPE_DEVICE);

    if (is_active && is_device) {
        if (is_p1) {
             current_m = 0.0;
        } else {
            double t = (double)step * param::dt;
            current_m = current_m + 0.5 * current_m * fabs(sin(t / 6000.0));
        }
    }

    // --- Part 2: Compute Forces and Update State ---
    double my_qx = 0.0, my_qy = 0.0, my_qz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double my_vx = 0.0, my_vy = 0.0, my_vz = 0.0;

    if (is_active) {
        my_qx = d_qx[i];
        my_qy = d_qy[i];
        my_qz = d_qz[i];
        my_vx = d_vx[i];
        my_vy = d_vy[i];
        my_vz = d_vz[i];
    }

    __shared__ double s_data[4 * BLOCK_SIZE];

    const double G_eps_sq = param::G;
    const double eps_sq = param::eps * param::eps;

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int idx = tile * BLOCK_SIZE + tid;
        
        // Load data from Global Memory (for all bodies j in the universe)
        if (idx < n) {
            s_data[0 * STRIDE + tid] = d_qx[idx];
            s_data[1 * STRIDE + tid] = d_qy[idx];
            s_data[2 * STRIDE + tid] = d_qz[idx];
            
            // Re-calculate mass for body j
            double mass_j = d_base_mass[idx];
            if (d_type[idx] == TYPE_DEVICE) {
                 if (is_p1) {
                    mass_j = 0.0;
                 } else {
                    double t = (double)step * param::dt;
                    mass_j = mass_j + 0.5 * mass_j * fabs(sin(t / 6000.0));
                 }
            }
            s_data[3 * STRIDE + tid] = mass_j;
        } else {
            s_data[3 * STRIDE + tid] = 0.0; 
        }
        __syncthreads();

        if (is_active) {
            #pragma unroll UNROLL_FACTOR
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int other_idx = tile * BLOCK_SIZE + j;
                if (other_idx >= n) break;

                double dx = s_data[0 * STRIDE + j] - my_qx;
                double dy = s_data[1 * STRIDE + j] - my_qy;
                double dz = s_data[2 * STRIDE + j] - my_qz;
                double mass_j_shared = s_data[3 * STRIDE + j];

                // Expose FMA for distance squared
                double dx2 = dx * dx;
                double dy2 = dy * dy;
                double dz2 = dz * dz;
                double dist_sq = dx2 + dy2;
                dist_sq += dz2;
                dist_sq += eps_sq;
                
                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                
                double f = G_eps_sq * mass_j_shared * dist_inv3;
                
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    if (is_active) {
        // Update velocity and position
        double v_new_x = my_vx + ax * param::dt;
        double v_new_y = my_vy + ay * param::dt;
        double v_new_z = my_vz + az * param::dt;
        
        d_vx[i] = v_new_x;
        d_vy[i] = v_new_y;
        d_vz[i] = v_new_z;
        
        d_qx[i] = my_qx + v_new_x * param::dt;
        d_qy[i] = my_qy + v_new_y * param::dt;
        d_qz[i] = my_qz + v_new_z * param::dt;
    }
}

__global__ void check_collision_single_kernel(
    const double* qx, const double* qy, const double* qz,
    int planet, int asteroid, int step,
    double* min_dist, int* hit_step) 
{
    if (threadIdx.x != 0) return;
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (min_dist) atomicMinDouble(min_dist, dist);
    if (hit_step) {
        if (dist < param::planet_radius) atomicCAS(hit_step, -2, step);
    }
}

// ---------------------------------------------------------
// HOST FUNCTIONS
// ---------------------------------------------------------
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& type, std::vector<int>& device_ids) {
    
    std::ifstream fin(filename);
    if (!fin.is_open()) exit(1);
    fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n);
    vx.resize(n); vy.resize(n); vz.resize(n);
    m.resize(n); type.resize(n);
    for (int i = 0; i < n; i++) {
        std::string t_str;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> t_str;
        if (t_str == "device") {
            type[i] = TYPE_DEVICE;
            device_ids.push_back(i);
        } else {
            type[i] = TYPE_NORMAL;
        }
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// Single GPU version for P1 and P2 
std::pair<double, int> run_single_sim_fast(
    int n, int planet, int asteroid, int n_steps,
    const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
    const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
    const std::vector<double>& h_m, const std::vector<int>& h_type,
    int device_id, bool is_p1) 
{
    HIP_CHECK(hipSetDevice(device_id));
    
    size_t sz_d = n * sizeof(double);
    size_t sz_i = n * sizeof(int);
    
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_base_mass;
    int *d_type;
    double *d_min_dist;
    int *d_hit_step;

    // FIX: Use simple bool pointer for device communication
    bool h_is_zero_mass = is_p1;
    bool *d_is_zero_mass;
    HIP_CHECK(hipMalloc(&d_is_zero_mass, sizeof(bool)));
    HIP_CHECK(hipMemcpy(d_is_zero_mass, &h_is_zero_mass, sizeof(bool), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&d_qx, sz_d)); HIP_CHECK(hipMalloc(&d_qy, sz_d)); HIP_CHECK(hipMalloc(&d_qz, sz_d));
    HIP_CHECK(hipMalloc(&d_vx, sz_d)); HIP_CHECK(hipMalloc(&d_vy, sz_d)); HIP_CHECK(hipMalloc(&d_vz, sz_d));
    HIP_CHECK(hipMalloc(&d_base_mass, sz_d)); 
    HIP_CHECK(hipMalloc(&d_type, sz_i));
    HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_qx, h_qx.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, h_qy.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, h_qz.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_base_mass, h_m.data(), sz_d, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), sz_i, hipMemcpyHostToDevice));
    
    double init_dist = std::numeric_limits<double>::infinity();
    int init_hit = -2;
    HIP_CHECK(hipMemcpy(d_min_dist, &init_dist, sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hit_step, &init_hit, sizeof(int), hipMemcpyHostToDevice));
    
    int threads = BLOCK_SIZE;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int s = 1; s <= n_steps; ++s) {
        // FUSED KERNEL CALL
        compute_and_update_all_single_kernel<<<blocks, threads>>>(
            d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_base_mass, d_type, n, s, d_is_zero_mass);
        
        check_collision_single_kernel<<<1, 1>>>(
            d_qx, d_qy, d_qz, planet, asteroid, s, d_min_dist, d_hit_step);
    }
    
    double h_min; int h_step;
    HIP_CHECK(hipMemcpy(&h_min, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
    HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_base_mass)); 
    HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_min_dist)); HIP_CHECK(hipFree(d_hit_step));
    HIP_CHECK(hipFree(d_is_zero_mass));
    
    return {h_min, h_step};
}

// Problem 3 Solver (Batched independent simulations) 
void run_batch_sim(
    int device_id_gpu, 
    const std::vector<int>& batch_targets,
    int n, int planet, int asteroid, int n_steps,
    const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
    const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
    const std::vector<double>& h_m, const std::vector<int>& h_type,
    int* h_missile_steps, int* h_planet_steps)
{
    int batch_size = batch_targets.size();
    if (batch_size == 0) return;

    HIP_CHECK(hipSetDevice(device_id_gpu));

    size_t sz_d_batch = n * batch_size * sizeof(double);
    size_t sz_d_single = n * sizeof(double);
    size_t sz_i_single = n * sizeof(int);

    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_base_mass;
    int *d_type, *d_targets, *d_missile_steps, *d_planet_steps;

    // Allocate batch arrays
    HIP_CHECK(hipMalloc(&d_qx, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_qy, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_qz, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vx, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vy, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vz, sz_d_batch));
    
    // Allocate single arrays (shared by all batches)
    HIP_CHECK(hipMalloc(&d_base_mass, sz_d_single));
    HIP_CHECK(hipMalloc(&d_type, sz_i_single));
    
    // Allocate results/targets
    HIP_CHECK(hipMalloc(&d_targets, batch_size * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_missile_steps, batch_size * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_planet_steps, batch_size * sizeof(int)));

    // 1. Copy initial state (1xN) for batch 0 only
    HIP_CHECK(hipMemcpy(d_qx, h_qx.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, h_qy.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, h_qz.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), sz_d_single, hipMemcpyHostToDevice));

    // 2. Launch broadcast kernel to replicate initial state (device-side copy)
    int threads = BLOCK_SIZE;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_single(blocks, 1);
    broadcast_kernel<<<grid_single, threads>>>(d_qx, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_qy, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_qz, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vx, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vy, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vz, n, batch_size);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy single-instance shared data
    HIP_CHECK(hipMemcpy(d_base_mass, h_m.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), sz_i_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_targets, batch_targets.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Initialize result arrays on host and copy to device
    int init_m = param::n_steps + 1;
    int init_p = -2;
    std::vector<int> host_init_m(batch_size, init_m);
    std::vector<int> host_init_p(batch_size, init_p);
    HIP_CHECK(hipMemcpy(d_missile_steps, host_init_m.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_planet_steps, host_init_p.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Kernel launch loop
    dim3 grid_batch(blocks, batch_size); 

    for(int s=1; s<=n_steps; ++s) {
        // FUSED KERNEL CALL
        compute_and_update_all_batched_kernel<<<grid_batch, threads>>>(
            d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_base_mass, d_type,
            n, s, d_targets, planet, d_missile_steps);

        check_collision_batched_kernel<<<dim3(1, batch_size), 1>>>(
            d_qx, d_qy, d_qz, planet, asteroid, s, n, d_planet_steps);
    }

    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy results back to host
    HIP_CHECK(hipMemcpy(h_missile_steps, d_missile_steps, batch_size * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_planet_steps, d_planet_steps, batch_size * sizeof(int), hipMemcpyDeviceToHost));

    // Free device memory
    HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
    HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_base_mass)); 
    HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_targets)); HIP_CHECK(hipFree(d_missile_steps)); HIP_CHECK(hipFree(d_planet_steps));
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type, device_ids;

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, device_ids);

    // Enable peer access
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

    std::pair<double, int> res1_out;
    std::pair<double, int> res2_out;
    
    // Parallelize P1 and P2
    std::thread t_p1([&]() {
        // P1: Run on GPU 0, is_p1 = true (devices disabled)
        res1_out = run_single_sim_fast(n, planet, asteroid, param::n_steps, 
                                       qx, qy, qz, vx, vy, vz, m, type, 0, true);
    });

    std::thread t_p2([&]() {
        // P2: Run on GPU 1, is_p1 = false (devices enabled)
        res2_out = run_single_sim_fast(n, planet, asteroid, param::n_steps, 
                                       qx, qy, qz, vx, vy, vz, m, type, 1, false);
    });

    t_p1.join();
    t_p2.join();

    auto res1 = res1_out;
    auto res2 = res2_out;
    
    int best_device_id = -1;
    double min_cost = 0.0;

    // Problem 3: Only run if collision happened in P2
    if (res2.second != -2 && !device_ids.empty()) {
        std::vector<int> batch0, batch1;
        for(size_t i=0; i<device_ids.size(); ++i) {
            if(i % 2 == 0) batch0.push_back(device_ids[i]);
            else batch1.push_back(device_ids[i]);
        }

        std::vector<int> m_steps0(batch0.size()), p_steps0(batch0.size());
        std::vector<int> m_steps1(batch1.size()), p_steps1(batch1.size());

        std::thread t0([&]() {
            if (!batch0.empty())
                run_batch_sim(0, batch0, n, planet, asteroid, param::n_steps, 
                              qx, qy, qz, vx, vy, vz, m, type, m_steps0.data(), p_steps0.data());
        });
        
        std::thread t1([&]() {
            if (!batch1.empty())
                run_batch_sim(1, batch1, n, planet, asteroid, param::n_steps, 
                              qx, qy, qz, vx, vy, vz, m, type, m_steps1.data(), p_steps1.data());
        });

        t0.join();
        t1.join();

        min_cost = std::numeric_limits<double>::infinity();

        auto check_results = [&](const std::vector<int>& targets, const std::vector<int>& m_steps, const std::vector<int>& p_steps) {
            for(size_t i=0; i<targets.size(); ++i) {
                int p_hit = p_steps[i];
                int m_hit = m_steps[i];
                
                if (p_hit == -2 && m_hit <= param::n_steps) { 
                    double cost = 1e5 + (m_hit) * param::dt * 1e3; 
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_device_id = targets[i];
                    }
                }
            }
        };

        check_results(batch0, m_steps0, p_steps0);
        check_results(batch1, m_steps1, p_steps1);
    }
    
    if (best_device_id == -1) {
        min_cost = 0;
    }

    write_output(argv[2], res1.first, res2.second, best_device_id, min_cost);
    return 0;
}