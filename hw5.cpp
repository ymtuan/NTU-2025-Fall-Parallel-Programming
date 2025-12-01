#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <map>

#define HIP_CHECK(err) \
    do { \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

struct SimParams {
    int n;
    int n_steps;
    double dt;
    double eps;
    double G;
    double planet_radius;
    double missile_speed;
    int planet_id;
    int asteroid_id;
};

__constant__ SimParams d_params;

const int TYPE_NORMAL = 0;
const int TYPE_DEVICE = 1;
const int CKPT_INTERVAL = 1000;
#define MAX_N 1024 
// Threshold: Small N runs on 1 block (low latency), Large N runs on many blocks (high throughput)
#define HYBRID_THRESHOLD 256 
#define LARGE_BLOCK_SIZE  64

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
// KERNEL 1: FUSED PARALLEL (Best for N > 256)
// Combines Mass Calc, Force, Update, and Checks into ONE kernel
// ---------------------------------------------------------
// ---------------------------------------------------------
// FINAL OPTIMIZED KERNEL: GHOST PARTICLES + PRE-MULTIPLIED G
// Best balance of Occupancy and ALU throughput.
// ---------------------------------------------------------
__global__ void __launch_bounds__(256) step_kernel_large_fused(
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ mass_base, 
    const int* __restrict__ type,
    const double* __restrict__ sin_table,
    int step,
    bool is_p1,
    double* d_min_dist,
    int* d_hit_step,
    int* d_device_destroy_steps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_params.n;

    // 1. Planet Cache (Shared Mem)
    __shared__ double s_planet_pos[3];
    if (threadIdx.x == 0) {
        int pid = d_params.planet_id;
        s_planet_pos[0] = qx_in[pid];
        s_planet_pos[1] = qy_in[pid];
        s_planet_pos[2] = qz_in[pid];
    }

    // 2. Load Local State
    double my_qx = 0.0, my_qy = 0.0, my_qz = 0.0;
    double my_vx, my_vy, my_vz;
    
    if (i < n) {
        my_qx = qx_in[i]; my_qy = qy_in[i]; my_qz = qz_in[i];
        my_vx = vx_in[i]; my_vy = vy_in[i]; my_vz = vz_in[i];
    }
    // No else needed for pos, initialized to 0.

    double ax = 0.0, ay = 0.0, az = 0.0;
    
    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_m[256]; // Will store (Mass * G)

    double eps_sq = d_params.eps * d_params.eps;
    double G = d_params.G;
    double sin_val = sin_table[step];
    int tile_count = (n + 255) / 256;

    // 3. Main Force Loop
    for (int tile = 0; tile < tile_count; tile++) {
        int idx = tile * 256 + threadIdx.x;
        
        // --- Optimized Load Phase ---
        if (idx < n) {
            s_qx[threadIdx.x] = qx_in[idx];
            s_qy[threadIdx.x] = qy_in[idx];
            s_qz[threadIdx.x] = qz_in[idx];
            
            double m_val = mass_base[idx];
            if (type[idx] == TYPE_DEVICE) {
                 // Branchless-friendly selection
                 if (is_p1) m_val = 0.0;
                 else m_val += 0.5 * m_val * sin_val;
            }
            // OPTIMIZATION: Pre-multiply by G here
            // Saves 1 multiplication per interaction in the inner loop
            s_m[threadIdx.x] = m_val * G; 
        } else {
            // Padding (Ghosts)
            s_qx[threadIdx.x] = 0.0; s_qy[threadIdx.x] = 0.0; s_qz[threadIdx.x] = 0.0;
            s_m[threadIdx.x]  = 0.0; 
        }
        __syncthreads();

        // --- Inner Loop (Branchless) ---
        // Unroll 16 is a safe sweet spot for register usage
        #pragma unroll 16
        for (int j = 0; j < 256; j++) {
            double dx = s_qx[j] - my_qx;
            double dy = s_qy[j] - my_qy;
            double dz = s_qz[j] - my_qz;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
            
            // Fast inverse sqrt
            double dist_inv = rsqrt(dist_sq);
            double dist_inv3 = dist_inv * dist_inv * dist_inv;
            
            // s_m[j] already contains (mass * G)
            double f = s_m[j] * dist_inv3;
            
            ax += f * dx; 
            ay += f * dy; 
            az += f * dz;
        }
        __syncthreads();
    }

    if (i >= n) return;

    // 4. Update
    double dt = d_params.dt;
    double new_vx = my_vx + ax * dt;
    double new_vy = my_vy + ay * dt;
    double new_vz = my_vz + az * dt;
    
    vx_out[i] = new_vx; vy_out[i] = new_vy; vz_out[i] = new_vz;
    qx_out[i] = my_qx + new_vx * dt; 
    qy_out[i] = my_qy + new_vy * dt; 
    qz_out[i] = my_qz + new_vz * dt;

    // 5. Collision & Missile Checks
    // (Identical to previous correct version)
    if (i == 0) {
        int aid = d_params.asteroid_id;
        double dx = s_planet_pos[0] - qx_in[aid];
        double dy = s_planet_pos[1] - qy_in[aid];
        double dz = s_planet_pos[2] - qz_in[aid];
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        atomicMinDouble(d_min_dist, dist);
        if (dist < d_params.planet_radius) atomicCAS(d_hit_step, -2, step);
    }

    if (!is_p1 && d_device_destroy_steps && type[i] == TYPE_DEVICE) {
        if (d_device_destroy_steps[i] == -1) {
            double dx = qx_in[i] - s_planet_pos[0];
            double dy = qy_in[i] - s_planet_pos[1];
            double dz = qz_in[i] - s_planet_pos[2];
            double m_dist = (double)step * dt * d_params.missile_speed;
            if (m_dist * m_dist > dx*dx + dy*dy + dz*dz) 
                d_device_destroy_steps[i] = step;
        }
    }
}

// ---------------------------------------------------------
// KERNEL 2: FUSED INTERVAL (Best for N <= 256)
// Runs entire time-loop in registers/shared mem
// ---------------------------------------------------------
__global__ void run_interval_kernel_single_block(
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ mass_base, 
    const int* __restrict__ type,
    const double* __restrict__ sin_table,
    int start_step,
    int n_steps_to_run,
    bool is_p1,
    double* d_min_dist,
    int* d_hit_step,
    int* d_device_destroy_steps
) {
    int i = threadIdx.x;
    int n = d_params.n;
    
    double my_qx = (i < n) ? qx_in[i] : 0.0;
    double my_qy = (i < n) ? qy_in[i] : 0.0;
    double my_qz = (i < n) ? qz_in[i] : 0.0;
    double my_vx = (i < n) ? vx_in[i] : 0.0;
    double my_vy = (i < n) ? vy_in[i] : 0.0;
    double my_vz = (i < n) ? vz_in[i] : 0.0;
    double my_m_base = (i < n) ? mass_base[i] : 0.0;
    int my_type    = (i < n) ? type[i] : TYPE_NORMAL;

    __shared__ double s_qx[MAX_N];
    __shared__ double s_qy[MAX_N];
    __shared__ double s_qz[MAX_N];
    __shared__ double s_m[MAX_N];

    double eps_sq = d_params.eps * d_params.eps;
    double G = d_params.G;
    double dt = d_params.dt;

    for (int k = 0; k < n_steps_to_run; k++) {
        int current_step = start_step + k;
        
        if (i < n) {
            s_qx[i] = my_qx; s_qy[i] = my_qy; s_qz[i] = my_qz;
            double m_val = my_m_base;
            if (my_type == TYPE_DEVICE) {
                if (is_p1) m_val = 0.0;
                else m_val = m_val + 0.5 * m_val * sin_table[current_step];
            }
            s_m[i] = m_val;
        } else { s_m[i] = 0.0; }
        __syncthreads();

        if (i < n) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            #pragma unroll 16
            for (int j = 0; j < n; j++) {
                double dx = s_qx[j] - my_qx;
                double dy = s_qy[j] - my_qy;
                double dz = s_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                double f = G * s_m[j] * dist_inv3;
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            my_vx += ax * dt; my_vy += ay * dt; my_vz += az * dt;
            my_qx += my_vx * dt; my_qy += my_vy * dt; my_qz += my_vz * dt;

            // Fused Collision Check
            if (i == 0) {
                int pid = d_params.planet_id; int aid = d_params.asteroid_id;
                double dx = s_qx[pid] - s_qx[aid]; double dy = s_qy[pid] - s_qy[aid]; double dz = s_qz[pid] - s_qz[aid];
                double dist = sqrt(dx*dx + dy*dy + dz*dz);
                atomicMinDouble(d_min_dist, dist);
                if (dist < d_params.planet_radius) {
                    int old = atomicCAS(d_hit_step, -2, current_step);
                    if (old != -2) atomicMin(d_hit_step, current_step);
                }
            }
            // Fused Missile Check
            if (!is_p1 && d_device_destroy_steps && my_type == TYPE_DEVICE) {
                 int pid = d_params.planet_id;
                 double dx = s_qx[i] - s_qx[pid]; double dy = s_qy[i] - s_qy[pid]; double dz = s_qz[i] - s_qz[pid];
                 double m_dist = (double)current_step * dt * d_params.missile_speed;
                 if (m_dist * m_dist > (dx*dx + dy*dy + dz*dz)) {
                     int old = d_device_destroy_steps[i];
                     if (old == -1) d_device_destroy_steps[i] = current_step;
                 }
            }
        }
        __syncthreads();
    }

    if (i < n) {
        qx_out[i] = my_qx; qy_out[i] = my_qy; qz_out[i] = my_qz;
        vx_out[i] = my_vx; vy_out[i] = my_vy; vz_out[i] = my_vz;
    }
}

// ---------------------------------------------------------
// P3 BATCH KERNEL (Unchanged)
// ---------------------------------------------------------
__global__ void run_p3_batch_kernel(
    const double* __restrict__ ckpt_data, const int* __restrict__ tasks_info, int* __restrict__ results,           
    const double* __restrict__ mass_base, const int* __restrict__ type, const double* __restrict__ sin_table,
    int n_tasks, int n_bodies
) {
    int task_idx = blockIdx.x;
    if (task_idx >= n_tasks) return;
    int i = threadIdx.x;

    int target_id = tasks_info[task_idx * 3 + 0];
    int destroy_step = tasks_info[task_idx * 3 + 1];
    int ckpt_idx = tasks_info[task_idx * 3 + 2];
    int offset = ckpt_idx * 6 * n_bodies;
    
    double my_qx = (i < n_bodies) ? ckpt_data[offset + 0*n_bodies + i] : 0.0;
    double my_qy = (i < n_bodies) ? ckpt_data[offset + 1*n_bodies + i] : 0.0;
    double my_qz = (i < n_bodies) ? ckpt_data[offset + 2*n_bodies + i] : 0.0;
    double my_vx = (i < n_bodies) ? ckpt_data[offset + 3*n_bodies + i] : 0.0;
    double my_vy = (i < n_bodies) ? ckpt_data[offset + 4*n_bodies + i] : 0.0;
    double my_vz = (i < n_bodies) ? ckpt_data[offset + 5*n_bodies + i] : 0.0;
    double my_m_base = (i < n_bodies) ? mass_base[i] : 0.0;
    int my_type = (i < n_bodies) ? type[i] : TYPE_NORMAL;

    int current_step = ckpt_idx * CKPT_INTERVAL;
    int end_step = d_params.n_steps;

    __shared__ double s_qx[MAX_N]; __shared__ double s_qy[MAX_N]; __shared__ double s_qz[MAX_N];
    __shared__ double s_m[MAX_N]; __shared__ int s_hit;

    if (i == 0) s_hit = -2;
    __syncthreads();

    double eps_sq = d_params.eps * d_params.eps; double G = d_params.G; double dt = d_params.dt;

    for (int s = current_step; s <= end_step; s++) {
        if (i < n_bodies) {
            s_qx[i] = my_qx; s_qy[i] = my_qy; s_qz[i] = my_qz;
            double m_val = my_m_base;
            if (my_type == TYPE_DEVICE) {
                if (i == target_id && s >= destroy_step) m_val = 0.0;
                else m_val = m_val + 0.5 * m_val * sin_table[s];
            }
            s_m[i] = m_val;
        } else { s_m[i] = 0.0; }
        __syncthreads();

        if (i == 0) {
            int pid = d_params.planet_id; int aid = d_params.asteroid_id;
            double dx = s_qx[pid] - s_qx[aid]; double dy = s_qy[pid] - s_qy[aid]; double dz = s_qz[pid] - s_qz[aid];
            if ((dx*dx + dy*dy + dz*dz) < (d_params.planet_radius * d_params.planet_radius)) { if (s_hit == -2) s_hit = s; }
        }
        __syncthreads();
        if (s_hit != -2) break;

        if (i < n_bodies) {
            double ax = 0.0, ay = 0.0, az = 0.0;
            #pragma unroll 16
            for (int j = 0; j < n_bodies; j++) {
                double dx = s_qx[j] - my_qx; double dy = s_qy[j] - my_qy; double dz = s_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                double f = G * s_m[j] * dist_inv3;
                ax += f * dx; ay += f * dy; az += f * dz;
            }
            my_vx += ax * dt; my_vy += ay * dt; my_vz += az * dt;
            my_qx += my_vx * dt; my_qy += my_vy * dt; my_qz += my_vz * dt;
        }
        __syncthreads();
    }
    if (i == 0) results[task_idx] = s_hit;
}

// ---------------------------------------------------------
// SIMULATOR
// ---------------------------------------------------------
class Simulator {
    int device_id; int n; SimParams params;
    double *d_qx[2], *d_qy[2], *d_qz[2], *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_mass; int *d_type; double *d_sin_table;
    double *d_min_dist; int *d_hit_step; int *d_device_destroy_steps;
    double *d_flat_ckpts; int *d_tasks; int *d_results;

public:
    Simulator(int dev_id, const SimParams& p, const std::vector<double>& h_m, const std::vector<int>& h_t, const std::vector<double>& h_sin) 
        : device_id(dev_id), n(p.n), params(p) {
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpyToSymbol(d_params, &params, sizeof(SimParams)));
        size_t sz_d = n * sizeof(double);
        for(int i=0; i<2; i++) {
            HIP_CHECK(hipMalloc(&d_qx[i], sz_d)); HIP_CHECK(hipMalloc(&d_qy[i], sz_d)); HIP_CHECK(hipMalloc(&d_qz[i], sz_d));
            HIP_CHECK(hipMalloc(&d_vx[i], sz_d)); HIP_CHECK(hipMalloc(&d_vy[i], sz_d)); HIP_CHECK(hipMalloc(&d_vz[i], sz_d));
        }
        HIP_CHECK(hipMalloc(&d_mass, sz_d)); HIP_CHECK(hipMemcpy(d_mass, h_m.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_type, n * sizeof(int))); HIP_CHECK(hipMemcpy(d_type, h_t.data(), n * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_sin_table, (params.n_steps + 1) * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_sin_table, h_sin.data(), (params.n_steps + 1) * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double))); HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_device_destroy_steps, n * sizeof(int)));
    }

    std::pair<double, int> run_main(const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
        const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
        bool is_p1, std::vector<double>* flat_checkpoints = nullptr, std::vector<int>* device_destroy_steps_out = nullptr) 
    {
        HIP_CHECK(hipSetDevice(device_id));
        size_t sz_d = n * sizeof(double);
        HIP_CHECK(hipMemcpy(d_qx[0], h_qx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[0], h_qy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[0], h_qz.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[0], h_vx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[0], h_vy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[0], h_vz.data(), sz_d, hipMemcpyHostToDevice));

        double inf = std::numeric_limits<double>::infinity(); int no_hit = -2;
        HIP_CHECK(hipMemcpy(d_min_dist, &inf, sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_hit_step, &no_hit, sizeof(int), hipMemcpyHostToDevice));

        if (!is_p1) {
            std::vector<int> init_dest(n, -1);
            HIP_CHECK(hipMemcpy(d_device_destroy_steps, init_dest.data(), n*sizeof(int), hipMemcpyHostToDevice));
        }

        int in = 0, out = 1;
        int n_ckpts = (params.n_steps / CKPT_INTERVAL) + 1;
        if (flat_checkpoints) {
            flat_checkpoints->resize(n_ckpts * 6 * n);
            auto copy_state = [&](int buf_idx, int ckpt_idx) {
                int base = ckpt_idx * 6 * n;
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 0*n], d_qx[buf_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 1*n], d_qy[buf_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 2*n], d_qz[buf_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 3*n], d_vx[buf_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 4*n], d_vy[buf_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 5*n], d_vz[buf_idx], sz_d, hipMemcpyDeviceToHost));
            };
            copy_state(0, 0); 
        }

        // HYBRID DISPATCH
        if (n <= HYBRID_THRESHOLD) {
            // SINGLE BLOCK FUSED INTERVAL (Fast for small N)
            for (int s = 0; s < params.n_steps; s += CKPT_INTERVAL) {
                int steps = std::min(CKPT_INTERVAL, params.n_steps - s);
                run_interval_kernel_single_block<<<1, 1024>>>(
                    d_qx[in], d_qy[in], d_qz[in], d_vx[in], d_vy[in], d_vz[in],
                    d_qx[out], d_qy[out], d_qz[out], d_vx[out], d_vy[out], d_vz[out],
                    d_mass, d_type, d_sin_table, s + 1, steps, is_p1, d_min_dist, d_hit_step, d_device_destroy_steps);
                
                if (flat_checkpoints) {
                    HIP_CHECK(hipDeviceSynchronize());
                    int ckpt_idx = (s + steps) / CKPT_INTERVAL;
                    int base = ckpt_idx * 6 * n;
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 0*n], d_qx[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 1*n], d_qy[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 2*n], d_qz[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 3*n], d_vx[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 4*n], d_vy[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 5*n], d_vz[out], sz_d, hipMemcpyDeviceToHost));
                }
                std::swap(in, out);
            }
        } else {
            // MULTI BLOCK FUSED (High Throughput for Large N)
            int blocks = (n + 255) / 256;
            for (int s = 1; s <= params.n_steps; s++) {
                step_kernel_large_fused<<<blocks, 256>>>(
                    d_qx[in], d_qy[in], d_qz[in], d_vx[in], d_vy[in], d_vz[in],
                    d_qx[out], d_qy[out], d_qz[out], d_vx[out], d_vy[out], d_vz[out],
                    d_mass, d_type, d_sin_table, s, is_p1, d_min_dist, d_hit_step, d_device_destroy_steps);
                
                if (flat_checkpoints && (s % CKPT_INTERVAL == 0)) {
                    HIP_CHECK(hipDeviceSynchronize());
                    int ckpt_idx = s / CKPT_INTERVAL;
                    int base = ckpt_idx * 6 * n;
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 0*n], d_qx[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 1*n], d_qy[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 2*n], d_qz[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 3*n], d_vx[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 4*n], d_vy[out], sz_d, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(&(*flat_checkpoints)[base + 5*n], d_vz[out], sz_d, hipMemcpyDeviceToHost));
                }
                std::swap(in, out);
            }
        }

        double h_min; int h_hit;
        HIP_CHECK(hipMemcpy(&h_min, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&h_hit, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
        if (device_destroy_steps_out) {
            device_destroy_steps_out->resize(n);
            HIP_CHECK(hipMemcpy(device_destroy_steps_out->data(), d_device_destroy_steps, n*sizeof(int), hipMemcpyDeviceToHost));
        }
        return {h_min, h_hit};
    }

    void run_batch_p3(const std::vector<double>& flat_checkpoints, const std::vector<int>& tasks_info, std::vector<int>& results_out) {
        if (tasks_info.empty()) return;
        HIP_CHECK(hipSetDevice(device_id));
        int n_tasks = tasks_info.size() / 3;
        results_out.resize(n_tasks);
        size_t sz_ckpts = flat_checkpoints.size() * sizeof(double);
        HIP_CHECK(hipMalloc(&d_flat_ckpts, sz_ckpts)); HIP_CHECK(hipMemcpy(d_flat_ckpts, flat_checkpoints.data(), sz_ckpts, hipMemcpyHostToDevice));
        size_t sz_tasks = tasks_info.size() * sizeof(int);
        HIP_CHECK(hipMalloc(&d_tasks, sz_tasks)); HIP_CHECK(hipMemcpy(d_tasks, tasks_info.data(), sz_tasks, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_results, n_tasks * sizeof(int)));
        run_p3_batch_kernel<<<n_tasks, 1024>>>(d_flat_ckpts, d_tasks, d_results, d_mass, d_type, d_sin_table, n_tasks, n);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(results_out.data(), d_results, n_tasks * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_flat_ckpts)); HIP_CHECK(hipFree(d_tasks)); HIP_CHECK(hipFree(d_results));
    }
};

void read_input(const char* filename, int& n, int& planet, int& asteroid, std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz, std::vector<double>& m, std::vector<int>& type, std::vector<int>& device_ids) {
    std::ifstream fin(filename); if (!fin.is_open()) exit(1);
    fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n); vx.resize(n); vy.resize(n); vz.resize(n); m.resize(n); type.resize(n);
    for (int i = 0; i < n; i++) {
        std::string t_str; fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> t_str;
        if (t_str == "device") { type[i] = TYPE_DEVICE; device_ids.push_back(i); } else type[i] = TYPE_NORMAL;
    }
}
void write_output(const char* filename, double min_dist, int hit_time_step, int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist << '\n' << hit_time_step << '\n' << gravity_device_id << ' ' << missile_cost << '\n';
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m; std::vector<int> type, device_ids;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, device_ids);
    SimParams params = {n, 200000, 60.0, 1e-3, 6.674e-11, 1e7, 1e6, planet, asteroid};

    // Pre-calculate Sin Table
    std::vector<double> sin_table(params.n_steps + 1);
    for(int s=0; s<=params.n_steps; ++s) {
        double t = (double)s * params.dt;
        sin_table[s] = fabs(sin(t / 6000.0));
    }

    HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

    Simulator sim0(0, params, m, type, sin_table); Simulator sim1(1, params, m, type, sin_table);
    std::pair<double, int> res_p2, res_p1;
    std::vector<double> checkpoints; std::vector<int> destroy_steps;

    std::thread t1([&]() { res_p1 = sim1.run_main(qx, qy, qz, vx, vy, vz, true); });
    std::thread t2([&]() { res_p2 = sim0.run_main(qx, qy, qz, vx, vy, vz, false, &checkpoints, &destroy_steps); });
    t1.join(); t2.join();

    int best_id = -1; double min_cost = std::numeric_limits<double>::infinity();
    if (res_p2.second != -2 && !device_ids.empty()) {
        std::vector<int> tasks0, tasks1;
        for (int id : device_ids) {
            int step = destroy_steps[id];
            if (step != -1 && step <= params.n_steps) {
                int ckpt = (step / CKPT_INTERVAL);
                if (tasks0.size() <= tasks1.size()) { tasks0.push_back(id); tasks0.push_back(step); tasks0.push_back(ckpt); }
                else { tasks1.push_back(id); tasks1.push_back(step); tasks1.push_back(ckpt); }
            }
        }
        std::vector<int> res0, res1;
        std::thread w0([&]() { sim0.run_batch_p3(checkpoints, tasks0, res0); });
        std::thread w1([&]() { sim1.run_batch_p3(checkpoints, tasks1, res1); });
        w0.join(); w1.join();
        auto process = [&](const std::vector<int>& tasks, const std::vector<int>& res) {
            for(size_t i=0; i<res.size(); i++) {
                if (res[i] == -2) {
                    double cost = 1e5 + (double)tasks[i*3+1] * params.dt * 1e3;
                    if (cost < min_cost) { min_cost = cost; best_id = tasks[i*3+0]; }
                }
            }
        };
        process(tasks0, res0); process(tasks1, res1);
    }
    if (best_id == -1) min_cost = 0;
    write_output(argv[2], res_p1.first, res_p2.second, best_id, min_cost);
    return 0;
}