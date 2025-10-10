#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image img(input_img);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);

    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);


    /////////////////////////////////////////////////////////////
    // The following code is for the validation
    // You can not change the logic of the following code, because it is used for judge system
    std::ofstream ofs(output_txt);
    if (!ofs) {
        std::cerr << "Failed to open " << output_txt << " for writing.\n";
    } else {
        ofs << kps.size() << "\n";
        for (const auto& kp : kps) {
            ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
            for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                ofs << " " << static_cast<int>(kp.descriptor[i]);
            }
            ofs << "\n";
        }
        ofs.close();
    }

    Image result = draw_keypoints(img, kps);
    result.save(output_img);
    /////////////////////////////////////////////////////////////

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms\n";
    
    std::cout << "Found " << kps.size() << " keypoints.\n";
    return 0;
}