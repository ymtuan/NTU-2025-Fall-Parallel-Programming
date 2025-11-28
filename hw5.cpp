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

// ---------------------------------------------------------
// CONSTANTS & PARAMETERS
// ---------------------------------------------------------
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

// Constant memory for read-only parameters
__constant__ SimParams d_params;

const int TYPE_NORMAL = 0;
const int TYPE_DEVICE = 1;
const int CKPT_INTERVAL = 1000; // Save state every 1000 steps

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------
struct Snapshot {
    std::vector<double> qx, qy, qz;
    std::vector<double> vx, vy, vz;
    // We don't need to save mass/type in snapshot as they are derived or constant
    int step;
};

// ---------------------------------------------------------
// KERNELS
// ---------------------------------------------------------

// Optimized Force Calculation with Shared Memory Tiling and Double Buffering
__global__ void step_kernel(
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ mass, 
    const int* __restrict__ type,
    int step,
    bool is_p1, // Problem 1 mode (devices have 0 mass)
    int destroy_target_id = -1, // For P3: ID of device to destroy
    int destroy_step = -1 // For P3: Step to destroy it
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_params.n;
    if (i >= n) return;

    // 1. Load Body i data
    double my_qx = qx_in[i];
    double my_qy = qy_in[i];
    double my_qz = qz_in[i];
    double my_vx = vx_in[i];
    double my_vy = vy_in[i];
    double my_vz = vz_in[i];

    // 2. Compute Force
    double ax = 0.0, ay = 0.0, az = 0.0;
    
    __shared__ double s_qx[256];
    __shared__ double s_qy[256];
    __shared__ double s_qz[256];
    __shared__ double s_m[256];

    double eps_sq = d_params.eps * d_params.eps;
    double G = d_params.G;

    for (int tile = 0; tile < (n + 255) / 256; tile++) {
        int idx = tile * 256 + threadIdx.x;
        
        // Cooperative load into shared memory
        if (idx < n) {
            s_qx[threadIdx.x] = qx_in[idx];
            s_qy[threadIdx.x] = qy_in[idx];
            s_qz[threadIdx.x] = qz_in[idx];
            
            // Handle Mass Fluctuation or Destruction Logic Here
            double m_val = mass[idx];
            if (type[idx] == TYPE_DEVICE) {
                if (is_p1) {
                    m_val = 0.0;
                } else {
                    // Check destruction for P3
                    if (idx == destroy_target_id && step >= destroy_step) {
                        m_val = 0.0;
                    } else {
                        // Fluctuation formula
                        double t = (double)step * d_params.dt;
                        m_val = m_val + 0.5 * m_val * fabs(sin(t / 6000.0));
                    }
                }
            }
            s_m[threadIdx.x] = m_val;
        } else {
            s_m[threadIdx.x] = 0.0;
        }
        __syncthreads();

        // Tile computation
        #pragma unroll 8
        for (int j = 0; j < 256; j++) {
            int j_idx = tile * 256 + j;
            if (j_idx >= n) break;

            double dx = s_qx[j] - my_qx;
            double dy = s_qy[j] - my_qy;
            double dz = s_qz[j] - my_qz;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
            double dist_inv3 = rsqrt(dist_sq * dist_sq * dist_sq);
            double f = G * s_m[j] * dist_inv3;

            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        __syncthreads();
    }

    // 3. Update State (Euler)
    double dt = d_params.dt;
    double new_vx = my_vx + ax * dt;
    double new_vy = my_vy + ay * dt;
    double new_vz = my_vz + az * dt;

    vx_out[i] = new_vx;
    vy_out[i] = new_vy;
    vz_out[i] = new_vz;

    qx_out[i] = my_qx + new_vx * dt;
    qy_out[i] = my_qy + new_vy * dt;
    qz_out[i] = my_qz + new_vz * dt;
}

// Distance Check Kernel
__global__ void check_hit_kernel(
    const double* qx, const double* qy, const double* qz,
    int step,
    double* d_min_dist,
    int* d_hit_step
) {
    if (threadIdx.x != 0) return;
    int pid = d_params.planet_id;
    int aid = d_params.asteroid_id;

    double dx = qx[pid] - qx[aid];
    double dy = qy[pid] - qy[aid];
    double dz = qz[pid] - qz[aid];
    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    // Update min dist
    if (d_min_dist) {
        unsigned long long* addr_as_ull = (unsigned long long*)d_min_dist;
        unsigned long long old = *addr_as_ull, assumed;
        do {
            assumed = old;
            if (__longlong_as_double(assumed) <= dist) break;
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(dist));
        } while (assumed != old);
    }

    // Update hit step
    if (dist < d_params.planet_radius) {
        atomicCAS(d_hit_step, -2, step);
    }
}

// Missile Logic Kernel: Check which devices are destroyable at this step
// This runs during P2 to pre-calculate P3 tasks
__global__ void check_missile_kernel(
    const double* qx, const double* qy, const double* qz,
    const int* type,
    int step,
    int* d_device_destroy_step
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_params.n) return;

    if (type[i] == TYPE_DEVICE) {
        // If already recorded a destroy step, don't overwrite with a later one
        // (We want the earliest possible valid hit time to minimize cost if we pick this device)
        if (d_device_destroy_step[i] != -1) return;

        int pid = d_params.planet_id;
        double px = qx[pid];
        double py = qy[pid];
        double pz = qz[pid];

        double dx = qx[i] - px;
        double dy = qy[i] - py;
        double dz = qz[i] - pz;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        double missile_travel = (double)step * d_params.dt * d_params.missile_speed;
        
        if (missile_travel * missile_travel > dist_sq) {
            d_device_destroy_step[i] = step;
        }
    }
}

// ---------------------------------------------------------
// HOST LOGIC
// ---------------------------------------------------------

class Simulator {
    int device_id;
    int n;
    SimParams params;
    
    // Double buffers
    double *d_qx[2], *d_qy[2], *d_qz[2];
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_mass;
    int *d_type;
    
    // Result buffers
    double *d_min_dist;
    int *d_hit_step;
    int *d_device_destroy_steps;

public:
    Simulator(int dev_id, const SimParams& p, 
              const std::vector<double>& h_m, const std::vector<int>& h_t) 
        : device_id(dev_id), n(p.n), params(p) 
    {
        HIP_CHECK(hipSetDevice(device_id));
        
        // Copy params to constant memory
        HIP_CHECK(hipMemcpyToSymbol(d_params, &params, sizeof(SimParams)));

        size_t sz_d = n * sizeof(double);
        for(int i=0; i<2; i++) {
            HIP_CHECK(hipMalloc(&d_qx[i], sz_d));
            HIP_CHECK(hipMalloc(&d_qy[i], sz_d));
            HIP_CHECK(hipMalloc(&d_qz[i], sz_d));
            HIP_CHECK(hipMalloc(&d_vx[i], sz_d));
            HIP_CHECK(hipMalloc(&d_vy[i], sz_d));
            HIP_CHECK(hipMalloc(&d_vz[i], sz_d));
        }
        HIP_CHECK(hipMalloc(&d_mass, sz_d));
        HIP_CHECK(hipMemcpy(d_mass, h_m.data(), sz_d, hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_type, n * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_type, h_t.data(), n * sizeof(int), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
        HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_device_destroy_steps, n * sizeof(int)));
    }

    // Run P2 (Normal) + P1 (No Devices) logic
    std::pair<double, int> run_main(
        const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
        const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
        bool is_p1,
        std::map<int, Snapshot>* checkpoints = nullptr,
        std::vector<int>* device_destroy_steps_out = nullptr
    ) {
        HIP_CHECK(hipSetDevice(device_id));

        // Init buffers
        size_t sz_d = n * sizeof(double);
        HIP_CHECK(hipMemcpy(d_qx[0], h_qx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[0], h_qy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[0], h_qz.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[0], h_vx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[0], h_vy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[0], h_vz.data(), sz_d, hipMemcpyHostToDevice));

        double inf = std::numeric_limits<double>::infinity();
        int no_hit = -2;
        HIP_CHECK(hipMemcpy(d_min_dist, &inf, sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_hit_step, &no_hit, sizeof(int), hipMemcpyHostToDevice));

        if (!is_p1 && device_destroy_steps_out) {
            std::vector<int> init_dest(n, -1);
            HIP_CHECK(hipMemcpy(d_device_destroy_steps, init_dest.data(), n*sizeof(int), hipMemcpyHostToDevice));
        }

        int blocks = (n + 255) / 256;
        int in_idx = 0;
        int out_idx = 1;

        // Save initial checkpoint (Step 0)
        if (checkpoints) {
            Snapshot s; s.step = 0; s.qx = h_qx; s.qy = h_qy; s.qz = h_qz; s.vx = h_vx; s.vy = h_vy; s.vz = h_vz;
            (*checkpoints)[0] = s;
        }

        for (int s = 1; s <= params.n_steps; s++) {
            // Force & Update Kernel (Double Buffered)
            step_kernel<<<blocks, 256>>>(
                d_qx[in_idx], d_qy[in_idx], d_qz[in_idx],
                d_vx[in_idx], d_vy[in_idx], d_vz[in_idx],
                d_qx[out_idx], d_qy[out_idx], d_qz[out_idx],
                d_vx[out_idx], d_vy[out_idx], d_vz[out_idx],
                d_mass, d_type, s, is_p1
            );

            // Distance Check
            check_hit_kernel<<<1, 1>>>(d_qx[out_idx], d_qy[out_idx], d_qz[out_idx], s, d_min_dist, d_hit_step);

            // Missile Check (Only for P2)
            if (!is_p1 && device_destroy_steps_out) {
                check_missile_kernel<<<blocks, 256>>>(d_qx[out_idx], d_qy[out_idx], d_qz[out_idx], d_type, s, d_device_destroy_steps);
            }

            // Checkpoint (Only for P2)
            if (checkpoints && (s % CKPT_INTERVAL == 0)) {
                HIP_CHECK(hipDeviceSynchronize()); // Ensure step complete before copy
                Snapshot snap;
                snap.step = s;
                snap.qx.resize(n); snap.qy.resize(n); snap.qz.resize(n);
                snap.vx.resize(n); snap.vy.resize(n); snap.vz.resize(n);
                
                HIP_CHECK(hipMemcpy(snap.qx.data(), d_qx[out_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(snap.qy.data(), d_qy[out_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(snap.qz.data(), d_qz[out_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(snap.vx.data(), d_vx[out_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(snap.vy.data(), d_vy[out_idx], sz_d, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(snap.vz.data(), d_vz[out_idx], sz_d, hipMemcpyDeviceToHost));
                (*checkpoints)[s] = snap;
            }

            // Swap buffers
            std::swap(in_idx, out_idx);
        }

        double h_min;
        int h_hit;
        HIP_CHECK(hipMemcpy(&h_min, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&h_hit, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));

        if (!is_p1 && device_destroy_steps_out) {
            device_destroy_steps_out->resize(n);
            HIP_CHECK(hipMemcpy(device_destroy_steps_out->data(), d_device_destroy_steps, n*sizeof(int), hipMemcpyDeviceToHost));
        }

        return {h_min, h_hit};
    }

    // Run Partial Simulation for P3 starting from Checkpoint
    int run_partial_p3(const Snapshot& start_state, int target_id, int destroy_step) {
        HIP_CHECK(hipSetDevice(device_id));

        size_t sz_d = n * sizeof(double);
        
        // Load checkpoint state
        HIP_CHECK(hipMemcpy(d_qx[0], start_state.qx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qy[0], start_state.qy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qz[0], start_state.qz.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vx[0], start_state.vx.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vy[0], start_state.vy.data(), sz_d, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vz[0], start_state.vz.data(), sz_d, hipMemcpyHostToDevice));

        // Reset hit
        int no_hit = -2;
        HIP_CHECK(hipMemcpy(d_hit_step, &no_hit, sizeof(int), hipMemcpyHostToDevice));

        int start_s = start_state.step;
        int blocks = (n + 255) / 256;
        int in_idx = 0;
        int out_idx = 1;

        for (int s = start_s + 1; s <= params.n_steps; s++) {
            step_kernel<<<blocks, 256>>>(
                d_qx[in_idx], d_qy[in_idx], d_qz[in_idx],
                d_vx[in_idx], d_vy[in_idx], d_vz[in_idx],
                d_qx[out_idx], d_qy[out_idx], d_qz[out_idx],
                d_vx[out_idx], d_vy[out_idx], d_vz[out_idx],
                d_mass, d_type, s, false, 
                target_id, destroy_step
            );

            // We pass nullptr for min_dist because P3 only cares about collision, not min dist
            check_hit_kernel<<<1, 1>>>(d_qx[out_idx], d_qy[out_idx], d_qz[out_idx], s, nullptr, d_hit_step);
            
            std::swap(in_idx, out_idx);
        }
        
        int h_hit;
        HIP_CHECK(hipMemcpy(&h_hit, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
        return h_hit;
    }
};

// ---------------------------------------------------------
// MAIN
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

int main(int argc, char** argv) {
    if (argc != 3) return 1;

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type, device_ids;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type, device_ids);

    SimParams params;
    params.n = n;
    params.n_steps = 200000;
    params.dt = 60.0;
    params.eps = 1e-3;
    params.G = 6.674e-11;
    params.planet_radius = 1e7;
    params.missile_speed = 1e6;
    params.planet_id = planet;
    params.asteroid_id = asteroid;

    std::pair<double, int> res_p2;
    std::pair<double, int> res_p1;
    std::map<int, Snapshot> checkpoints;
    std::vector<int> destroy_steps;

    // Fixed: Wrapped with HIP_CHECK to resolve compiler warnings
    HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipDeviceEnablePeerAccess(1, 0));
    HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

    Simulator sim0(0, params, m, type);
    Simulator sim1(1, params, m, type);

    std::thread t1([&]() {
        // Run P1 on GPU 1
        res_p1 = sim1.run_main(qx, qy, qz, vx, vy, vz, true);
    });

    std::thread t2([&]() {
        // Run P2 on GPU 0 with Checkpointing
        res_p2 = sim0.run_main(qx, qy, qz, vx, vy, vz, false, &checkpoints, &destroy_steps);
    });

    t1.join();
    t2.join();

    // Problem 3 Logic
    int best_id = -1;
    double min_cost = std::numeric_limits<double>::infinity();

    if (res_p2.second != -2 && !device_ids.empty()) {
        struct Task {
            int id;
            int destroy_step;
            int ckpt_step;
        };
        std::vector<Task> tasks;

        for (int id : device_ids) {
            int step = destroy_steps[id];
            if (step != -1 && step <= params.n_steps) {
                // Find closest checkpoint <= step
                // Since steps are 1, 2, ... and ckpts are 0, 1000, 2000...
                // (step / 1000) * 1000 gives the floor checkpoint
                int ckpt_step = (step / CKPT_INTERVAL) * CKPT_INTERVAL;
                tasks.push_back({id, step, ckpt_step});
            }
        }

        std::mutex mtx;
        
        auto worker = [&](Simulator* sim, std::vector<Task>& my_tasks) {
            for (const auto& t : my_tasks) {
                // Read from checkpoint map (safe because P2 is done writing to it)
                int hit = sim->run_partial_p3(checkpoints[t.ckpt_step], t.id, t.destroy_step);
                
                // If planet NOT hit, we found a candidate
                if (hit == -2) {
                    double cost = 1e5 + (double)t.destroy_step * params.dt * 1e3;
                    std::lock_guard<std::mutex> lock(mtx);
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_id = t.id;
                    } 
                }
            }
        };

        std::vector<Task> tasks0, tasks1;
        for (size_t i = 0; i < tasks.size(); i++) {
            if (i % 2 == 0) tasks0.push_back(tasks[i]);
            else tasks1.push_back(tasks[i]);
        }

        std::thread w0(worker, &sim0, std::ref(tasks0));
        std::thread w1(worker, &sim1, std::ref(tasks1));
        w0.join();
        w1.join();
    }

    if (best_id == -1) min_cost = 0;

    write_output(argv[2], res_p1.first, res_p2.second, best_id, min_cost);

    return 0;
}