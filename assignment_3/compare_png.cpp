#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip> // For std::fixed and std::setprecision
#include "lodepng/lodepng.h" // Assumes lodepng.h is in the lodepng subdirectory

// Function to compare two images loaded into memory
// Returns the percentage of matching pixels (all 4 channels must match)
double compare_images(const std::vector<unsigned char>& img1, unsigned w1, unsigned h1,
                      const std::vector<unsigned char>& img2, unsigned w2, unsigned h2) {
    // Check dimensions first
    if (w1 != w2 || h1 != h2) {
        std::cerr << "Warning: Image dimensions differ (" << w1 << "x" << h1 << " vs " << w2 << "x" << h2 << ")" << std::endl;
        return 0.0;
    }

    // Check if data size makes sense (expecting 4 bytes per pixel)
    size_t total_pixels = static_cast<size_t>(w1) * h1;
    size_t expected_bytes = total_pixels * 4;

    if (img1.size() != expected_bytes || img2.size() != expected_bytes) {
         std::cerr << "Warning: Image data size mismatch. Expected RGBA (4 bytes/pixel)." << std::endl;
         std::cerr << "Img1 bytes: " << img1.size() << ", Img2 bytes: " << img2.size() << ", Expected: " << expected_bytes << std::endl;
         // Proceed with comparison up to the smaller size if possible, but it indicates an issue.
         // For stricter checking, could return 0.0 here. Let's compare what we can.
    }

    size_t bytes_to_compare = std::min(img1.size(), img2.size());
     if (bytes_to_compare % 4 != 0) {
        std::cerr << "Warning: Number of bytes to compare is not a multiple of 4. Pixel data might be misaligned." << std::endl;
        // Adjust to the largest multiple of 4 less than or equal to bytes_to_compare
        bytes_to_compare -= (bytes_to_compare % 4);
    }


    long long matching_pixels = 0;
    size_t pixels_compared = 0;


    // Compare pixel data (RGBA)
    for (size_t i = 0; i < bytes_to_compare; i += 4) {
        pixels_compared++;
        // Check if all 4 channels (R, G, B, A) match
        if (img1[i] == img2[i] &&          // R
            img1[i + 1] == img2[i + 1] &&  // G
            img1[i + 2] == img2[i + 2] &&  // B
            img1[i + 3] == img2[i + 3]) {  // A
            matching_pixels++;
        }
    }

     if (pixels_compared != total_pixels) {
        std::cerr << "Warning: Compared " << pixels_compared << " pixels, but expected " << total_pixels << " based on dimensions." << std::endl;
    }


    // Calculate accuracy based on the number of pixels defined by dimensions
    return (total_pixels > 0) ? (static_cast<double>(matching_pixels) / total_pixels) * 100.0 : 100.0;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image1.png> <image2.png>" << std::endl;
        return 1; // Indicate error
    }

    const char* filename1 = argv[1];
    const char* filename2 = argv[2];

    std::vector<unsigned char> image1, image2;
    unsigned int width1, height1, width2, height2;

    // Decode image1
    // Use LCT_RGBA to ensure 4 channels, even if original is different
    unsigned error1 = lodepng::decode(image1, width1, height1, filename1, LCT_RGBA, 8);
    if (error1) {
        std::cerr << "Error decoding " << filename1 << ": " << lodepng_error_text(error1) << std::endl;
        // Output 0 accuracy on error to allow script to continue
        std::cout << std::fixed << std::setprecision(2) << 0.00 << std::endl;
        return 1; // Indicate failure to compare
    }

    // Decode image2
    unsigned error2 = lodepng::decode(image2, width2, height2, filename2, LCT_RGBA, 8);
    if (error2) {
        std::cerr << "Error decoding " << filename2 << ": " << lodepng_error_text(error2) << std::endl;
         // Output 0 accuracy on error
        std::cout << std::fixed << std::setprecision(2) << 0.00 << std::endl;
        return 1; // Indicate failure to compare
    }

    // Compare and get accuracy
    double accuracy = compare_images(image1, width1, height1, image2, width2, height2);

    // Print accuracy percentage to stdout, formatted to two decimal places
    std::cout << std::fixed << std::setprecision(2) << accuracy << std::endl;

    return 0; // Indicate successful comparison (even if accuracy is low)
}

