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
    
    // auto start = std::chrono::high_resolution_clock::now();
    
    Image img;
    
    if (world_rank == 0) {
        Image original_img(input_img);
        Image gray_img = original_img.channels == 1 ? original_img : rgb_to_grayscale(original_img);
        
        float base_sigma = SIGMA_MIN / MIN_PIX_DIST;
        Image base_img = gray_img.resize(gray_img.width * 2, gray_img.height * 2, Interpolation::BILINEAR);
        float sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0f);
        base_img = gaussian_blur(base_img, sigma_diff);
        
        std::vector<Image> octave_base_images;
        octave_base_images.push_back(base_img);
        for (int i = 1; i < N_OCT; ++i) {
            const Image& prev_base = octave_base_images.back();
            octave_base_images.push_back(prev_base.resize(prev_base.width/2, prev_base.height/2, Interpolation::NEAREST));
        }
        
        int octaves_per_process = N_OCT / world_size;
        for (int i = 1; i < world_size; ++i) {
            int start_octave = i * octaves_per_process;
            const Image& octave_base = octave_base_images[start_octave];
            int props[] = {octave_base.width, octave_base.height, octave_base.channels};
            MPI_Send(props, 3, MPI_INT, i , 0, MPI_COMM_WORLD);
            MPI_Send(octave_base.data, octave_base.size, MPI_FLOAT, i , 1, MPI_COMM_WORLD);
        }
        img = octave_base_images[0];
    }
    else {
        int props[3];
        MPI_Recv(props, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        img = Image(props[0], props[1], props[2]);
        MPI_Recv(img.data, img.size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting timer
    auto start = std::chrono::high_resolution_clock::now();
    

    // std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);
    std::vector<Keypoint> local_kps = find_keypoints_and_descriptors(img, world_rank, world_size);
      
    auto end = std::chrono::high_resolution_clock::now();
    
    const int ints_per_kp = 4 + (2 * sizeof(float) / sizeof(int)) + 128;
    int local_kp_count = local_kps.size();
    
    std::vector<int> local_kps_data(local_kp_count * ints_per_kp);
    for(int i = 0; i < local_kp_count; ++i) {
       const auto& kp = local_kps[i];
       int* offset = &local_kps_data[i * ints_per_kp];
       offset[0] = kp.i;
       offset[1] = kp.j;
       offset[2] = kp.octave;
       offset[3] = kp.scale;
       
       std::memcpy(&offset[4], &kp.x, sizeof(float));
       std::memcpy(&offset[4 + sizeof(float) / sizeof(int)], &kp.y, sizeof(float));
       
       for(int j = 0; j < 128; ++j) {
           offset[4 + 2 * sizeof(float) / sizeof(int) + j] = static_cast<int>(kp.descriptor[j]);
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
            std::memcpy(&kp.y, &offset[4 + sizeof(float) / sizeof(int)], sizeof(float));
            
            for(int j = 0; j < 128; ++j) {
                kp.descriptor[j] = static_cast<uint8_t>(offset[4 + 2 * sizeof(float) / sizeof(int) + j]);
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