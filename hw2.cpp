#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <math.h>
#include <numeric>
#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image img;
    int img_props[3];
    
    if (world_rank == 0) {
        img = Image(input_img);
        img_props[0] = img.width;
        img_props[1] = img.height;
        img_props[2] = img.channels;
    }
    
    MPI_Bcast(img_props, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        img = Image(img_props[0], img_props[1], img_props[2]);
    }
    
    MPI_Bcast(img.data, img.size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    

    Image gray_img = img.channels == 1 ? img : rgb_to_grayscale(img);
    

    std::vector<Keypoint> local_kps = find_keypoints_and_descriptors(gray_img, world_rank, world_size);
      
    
    // Define packet size: 4 ints + 2 floats (treated as ints) + 128 descriptor ints.
    const int ints_per_float = sizeof(float) / sizeof(int);
    const int ints_per_kp = 4 + (2 * ints_per_float) + 128;
    int local_kp_count = local_kps.size();
    
    std::vector<int> local_kps_data(local_kp_count * ints_per_kp);
    for(int i = 0; i < local_kp_count; ++i) {
         const auto& kp = local_kps[i];
         int* offset = &local_kps_data[i * ints_per_kp];
         offset[0] = kp.i;
         offset[1] = kp.j;
         offset[2] = kp.octave;
         offset[3] = kp.scale;
         
         // Safely copy float bytes into the integer array for transport.
         std::memcpy(&offset[4], &kp.x, sizeof(float));
         std::memcpy(&offset[4 + ints_per_float], &kp.y, sizeof(float));
  
         for(int j = 0; j < 128; ++j) {
             offset[4 + 2 * ints_per_float + j] = static_cast<int>(kp.descriptor[j]);
         }
    }
    
    std::vector<int> recv_counts(world_size, 0);
    MPI_Gather(&local_kp_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        // Calculate total keypoints and displacements for MPI_Gatherv
        int total_kps = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
        std::vector<Keypoint> all_kps;
        all_kps.reserve(total_kps);
        
        std::vector<int> recv_counts_data(world_size);
        std::vector<int> displacements(world_size);
        int current_displacement = 0;

        for (int i = 0; i < world_size; ++i) {
            recv_counts_data[i] = recv_counts[i] * ints_per_kp;
            displacements[i] = current_displacement;
            current_displacement += recv_counts_data[i];
        }

        std::vector<int> all_kps_data(total_kps * ints_per_kp);
        
        // Gather all serialized keypoint data from all processes
        MPI_Gatherv(local_kps_data.data(), local_kps_data.size(), MPI_INT,
                    all_kps_data.data(), recv_counts_data.data(), displacements.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);

        // De-serialize the data back into Keypoint objects
        for (int i = 0; i < total_kps; ++i) {
            Keypoint kp;
            int* offset = &all_kps_data[i * ints_per_kp];
            kp.i = offset[0];
            kp.j = offset[1];
            kp.octave = offset[2];
            kp.scale = offset[3];
            
            std::memcpy(&kp.x, &offset[4], sizeof(float));
            std::memcpy(&kp.y, &offset[4 + ints_per_float], sizeof(float));
            
            for(int j = 0; j < 128; ++j) {
                kp.descriptor[j] = static_cast<uint8_t>(offset[4 + 2 * ints_per_float + j]);
            }
            all_kps.push_back(kp);
        }

        // --- Output Section (only rank 0 executes this) ---
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << all_kps.size() << "\n";
            for (const auto& kp : all_kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }
        
        Image original_img(input_img);
        Image result = draw_keypoints(original_img, all_kps);
        result.save(output_img);
        // Image result = draw_keypoints(img, all_kps);
        // result.save(output_img);
        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << all_kps.size() << " keypoints.\n";
    } else {
        // Non-root processes send their data
        MPI_Gatherv(local_kps_data.data(), local_kps_data.size(), MPI_INT,
                    nullptr, nullptr, nullptr,
                    MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
    
}