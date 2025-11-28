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
#include <atomic>
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
// OPTIMIZATION: KERNEL FOR BROADCASTING INITIAL STATE
// ---------------------------------------------------------
// Copies the first N elements of d_arr (initial state) to the remaining (batch_size-1) * N elements.
__global__ void broadcast_kernel(double* d_arr, int n_bodies, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_bodies) return;

    double val = d_arr[tid];
    
    // Broadcast to all batch universes (starting from index 1)
    for (int i = 1; i < batch_size; ++i) {
        d_arr[i * n_bodies + tid] = val;
    }
}

// ---------------------------------------------------------
// CHECKPOINT STRUCTURE
// ---------------------------------------------------------
struct Checkpoint {
    std::vector<double> qx, qy, qz;
    std::vector<double> vx, vy, vz;
    int step;
};

// ---------------------------------------------------------
// BATCHED KERNELS (Problem 3) - UNCHANGED
// ---------------------------------------------------------
__global__ void update_mass_batched_kernel(
    const double* qx, const double* qy, const double* qz,
    const double* base_mass, const int* type, double* current_mass,
    int n, int step,
    const int* target_device_ids, 
    int planet_id, 
    int* missile_hit_steps)       
{
    int universe_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int offset = universe_idx * n;
    int global_i = offset + tid;

    double m = base_mass[tid]; 
    
    if (type[tid] == TYPE_DEVICE) {
        bool destroyed = false;
        int target_id = target_device_ids[universe_idx];

        if (target_id == tid) {
            double px = qx[offset + planet_id];
            double py = qy[offset + planet_id];
            double pz = qz[offset + planet_id];

            double dx = qx[global_i] - px;
            double dy = qy[global_i] - py;
            double dz = qz[global_i] - pz;
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            double missile_dist = (double)step * param::dt * param::missile_speed;
            if (missile_dist * missile_dist > dist_sq) {
                destroyed = true;
                m = 0.0;
                atomicMin(&missile_hit_steps[universe_idx], step);
            }
        }

        if (!destroyed) {
            double t = (double)step * param::dt;
            m = m + 0.5 * m * fabs(sin(t / 6000.0));
        }
    }
    current_mass[global_i] = m;
}

__global__ void compute_forces_batched_kernel(
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* current_mass,
    int n) 
{
    int universe_idx = blockIdx.y;
    int offset = universe_idx * n;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    bool is_active = (i < n);
    int global_i = offset + i;

    double my_qx = 0.0, my_qy = 0.0, my_qz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double my_vx = 0.0, my_vy = 0.0, my_vz = 0.0;

    if (is_active) {
        my_qx = qx[global_i];
        my_qy = qy[global_i];
        my_qz = qz[global_i];
        my_vx = vx[global_i];
        my_vy = vy[global_i];
        my_vz = vz[global_i];
    }

    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_mass[256];

    const double G_eps_sq = param::G;
    const double eps_sq = param::eps * param::eps;

    for (int tile = 0; tile < (n + 255) / 256; ++tile) {
        int idx = tile * 256 + tid;
        
        if (idx < n) {
            int load_idx = offset + idx;
            s_qx[tid] = qx[load_idx];
            s_qy[tid] = qy[load_idx];
            s_qz[tid] = qz[load_idx];
            s_mass[tid] = current_mass[load_idx];
        } else {
            s_mass[tid] = 0.0; 
        }
        __syncthreads();

        if (is_active) {
            #pragma unroll 8
            for (int j = 0; j < 256; ++j) {
                int other_local_idx = tile * 256 + j;
                if (other_local_idx >= n) break;
                
                double dx = s_qx[j] - my_qx;
                double dy = s_qy[j] - my_qy;
                double dz = s_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                
                double f = G_eps_sq * s_mass[j] * dist_inv3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    if (is_active) {
        double v_new_x = my_vx + ax * param::dt;
        double v_new_y = my_vy + ay * param::dt;
        double v_new_z = my_vz + az * param::dt;

        vx[global_i] = v_new_x;
        vy[global_i] = v_new_y;
        vz[global_i] = v_new_z;

        qx[global_i] = my_qx + v_new_x * param::dt;
        qy[global_i] = my_qy + v_new_y * param::dt;
        qz[global_i] = my_qz + v_new_z * param::dt;
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
// SINGLE SIMULATION KERNELS (Prob 1 & 2) - UNCHANGED
// ---------------------------------------------------------
__global__ void update_mass_single_kernel(
    const double* qx, const double* qy, const double* qz,
    const double* base_mass, const int* type, double* current_mass,
    int n, int step)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double m = base_mass[tid];
    if (type[tid] == TYPE_DEVICE) {
        double t = (double)step * param::dt;
        m = m + 0.5 * m * fabs(sin(t / 6000.0));
    }
    current_mass[tid] = m;
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

__global__ void compute_forces_split_kernel(
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* current_mass,
    int n, int offset, int my_n) 
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    bool is_active = (gid < my_n);
    int global_i = gid + offset;

    double my_qx = 0.0, my_qy = 0.0, my_qz = 0.0;
    double my_vx = 0.0, my_vy = 0.0, my_vz = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;

    if (is_active) {
        my_qx = qx[global_i];
        my_qy = qy[global_i];
        my_qz = qz[global_i];
        my_vx = vx[global_i];
        my_vy = vy[global_i];
        my_vz = vz[global_i];
    }

    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_mass[256];

    const double eps_sq = param::eps * param::eps;

    for (int tile = 0; tile < (n + 255) / 256; ++tile) {
        int idx = tile * 256 + tid;
        if (idx < n) {
            s_qx[tid] = qx[idx];
            s_qy[tid] = qy[idx];
            s_qz[tid] = qz[idx];
            s_mass[tid] = current_mass[idx];
        } else {
            s_mass[tid] = 0.0; 
        }
        __syncthreads();

        if (is_active) {
            #pragma unroll 8
            for (int j = 0; j < 256; ++j) {
                int other_idx = tile * 256 + j;
                if (other_idx >= n) break;
                double dx = s_qx[j] - my_qx;
                double dy = s_qy[j] - my_qy;
                double dz = s_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
                double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
                double f = param::G * s_mass[j] * dist_inv3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }
        __syncthreads();
    }

    if (is_active) {
        double v_new_x = my_vx + ax * param::dt;
        double v_new_y = my_vy + ay * param::dt;
        double v_new_z = my_vz + az * param::dt;
        vx[global_i] = v_new_x;
        vy[global_i] = v_new_y;
        vz[global_i] = v_new_z;
        qx[global_i] = my_qx + v_new_x * param::dt;
        qy[global_i] = my_qy + v_new_y * param::dt;
        qz[global_i] = my_qz + v_new_z * param::dt;
    }
}

// ---------------------------------------------------------
// HOST FUNCTIONS - UNCHANGED
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

struct GpuData {
    double *d_qx, *d_qy, *d_qz;
    double *d_vx, *d_vy, *d_vz;
    double *d_base_mass, *d_current_mass;
    int *d_type;
    double *d_min_dist;
    int *d_hit_step;
};

// Single GPU version for P1 and P2 
std::pair<double, int> run_single_sim_fast(
    int n, int planet, int asteroid, int n_steps,
    const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
    const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
    const std::vector<double>& h_m, const std::vector<int>& h_type,
    int device_id)
{
    HIP_CHECK(hipSetDevice(device_id));
    
    size_t sz_d = n * sizeof(double);
    size_t sz_i = n * sizeof(int);
    
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_base_mass, *d_current_mass;
    int *d_type;
    double *d_min_dist;
    int *d_hit_step;
    
    HIP_CHECK(hipMalloc(&d_qx, sz_d)); HIP_CHECK(hipMalloc(&d_qy, sz_d)); HIP_CHECK(hipMalloc(&d_qz, sz_d));
    HIP_CHECK(hipMalloc(&d_vx, sz_d)); HIP_CHECK(hipMalloc(&d_vy, sz_d)); HIP_CHECK(hipMalloc(&d_vz, sz_d));
    HIP_CHECK(hipMalloc(&d_base_mass, sz_d)); HIP_CHECK(hipMalloc(&d_current_mass, sz_d));
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
    
    int threads = 256;
    int blocks = (n + 255) / 256;
    
    for (int s = 1; s <= n_steps; ++s) {
        update_mass_single_kernel<<<blocks, threads>>>(
            d_qx, d_qy, d_qz, d_base_mass, d_type, d_current_mass, n, s);
        
        compute_forces_split_kernel<<<blocks, threads>>>(
            d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_current_mass, n, 0, n);
        
        check_collision_single_kernel<<<1, 1>>>(
            d_qx, d_qy, d_qz, planet, asteroid, s, d_min_dist, d_hit_step);
    }
    
    double h_min; int h_step;
    HIP_CHECK(hipMemcpy(&h_min, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_qx)); HIP_CHECK(hipFree(d_qy)); HIP_CHECK(hipFree(d_qz));
    HIP_CHECK(hipFree(d_vx)); HIP_CHECK(hipFree(d_vy)); HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_base_mass)); HIP_CHECK(hipFree(d_current_mass));
    HIP_CHECK(hipFree(d_type)); HIP_CHECK(hipFree(d_min_dist)); HIP_CHECK(hipFree(d_hit_step));
    
    return {h_min, h_step};
}

// Problem 3 Solver (Batched independent simulations) - OPTIMIZED
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

    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_cur_mass, *d_base_mass;
    int *d_type, *d_targets, *d_missile_steps, *d_planet_steps;

    // Allocate batch arrays
    HIP_CHECK(hipMalloc(&d_qx, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_qy, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_qz, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vx, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vy, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_vz, sz_d_batch));
    HIP_CHECK(hipMalloc(&d_cur_mass, sz_d_batch));
    
    // Allocate single arrays (shared by all batches)
    HIP_CHECK(hipMalloc(&d_base_mass, sz_d_single));
    HIP_CHECK(hipMalloc(&d_type, sz_i_single));
    
    // Allocate results/targets
    HIP_CHECK(hipMalloc(&d_targets, batch_size * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_missile_steps, batch_size * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_planet_steps, batch_size * sizeof(int)));

    // 1. Copy initial state (1xN) for batch 0 only (one H2D copy per array)
    HIP_CHECK(hipMemcpy(d_qx, h_qx.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, h_qy.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, h_qz.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, h_vx.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, h_vy.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, h_vz.data(), sz_d_single, hipMemcpyHostToDevice));

    // 2. Launch broadcast kernel to replicate initial state (device-side copy)
    int threads = 256;
    int blocks = (n + 255) / 256;
    dim3 grid_single(blocks, 1);
    broadcast_kernel<<<grid_single, threads>>>(d_qx, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_qy, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_qz, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vx, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vy, n, batch_size);
    broadcast_kernel<<<grid_single, threads>>>(d_vz, n, batch_size);
    HIP_CHECK(hipDeviceSynchronize()); // Must sync before the simulation starts

    // Copy single-instance shared data
    HIP_CHECK(hipMemcpy(d_base_mass, h_m.data(), sz_d_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_type, h_type.data(), sz_i_single, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_targets, batch_targets.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Initialize result arrays on host and copy to device
    int init_m = param::n_steps + 1; // Use a value outside the max step for "no hit"
    int init_p = -2;
    std::vector<int> host_init_m(batch_size, init_m);
    std::vector<int> host_init_p(batch_size, init_p);
    HIP_CHECK(hipMemcpy(d_missile_steps, host_init_m.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_planet_steps, host_init_p.data(), batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Kernel launch loop
    dim3 grid_batch(blocks, batch_size); 

    for(int s=1; s<=n_steps; ++s) {
        update_mass_batched_kernel<<<grid_batch, threads>>>(
            d_qx, d_qy, d_qz, d_base_mass, d_type, d_cur_mass,
            n, s, d_targets, planet, d_missile_steps);
        
        compute_forces_batched_kernel<<<grid_batch, threads>>>(
            d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_cur_mass, n);

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
    HIP_CHECK(hipFree(d_cur_mass)); HIP_CHECK(hipFree(d_base_mass)); HIP_CHECK(hipFree(d_type));
    HIP_CHECK(hipFree(d_targets)); HIP_CHECK(hipFree(d_missile_steps)); HIP_CHECK(hipFree(d_planet_steps));
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type, device_ids;

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, device_ids);

    // Enable peer access (important for P3 inter-GPU communication, though not explicitly used for data transfer here)
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

    // Prepare mass for P1 (no gravity devices)
    std::vector<double> m_zero = m;
    for(size_t i=0; i<n; ++i) if(type[i] == TYPE_DEVICE) m_zero[i] = 0.0;
    
    std::pair<double, int> res1_out;
    std::pair<double, int> res2_out;
    
    // OPTIMIZATION: Parallelize P1 and P2 across GPU 0 and GPU 1
    std::thread t_p1([&]() {
        res1_out = run_single_sim_fast(n, planet, asteroid, param::n_steps, 
                                       qx, qy, qz, vx, vy, vz, m_zero, type, 0);
    });

    std::thread t_p2([&]() {
        res2_out = run_single_sim_fast(n, planet, asteroid, param::n_steps, 
                                       qx, qy, qz, vx, vy, vz, m, type, 1);
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

        // P3 is already parallelized with threads and load-balanced
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
                
                // If collision was prevented (p_hit == -2) AND the missile actually launched (m_hit <= n_steps)
                if (p_hit == -2 && m_hit <= param::n_steps) { 
                    // Missile cost uses (step+1) for the formula, so if m_hit is the step, use (m_hit)
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