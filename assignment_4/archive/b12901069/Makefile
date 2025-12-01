NVFLAGS := -std=c++11 -O3 -rdc=true -arch=sm_70
PTXASFLAGS := -v --opt-level=3
TARGET := hw4

.PHONY: all
all: $(TARGET)

$(TARGET): hw4.cu sha256.cu
	nvcc $(NVFLAGS) -Xptxas="$(PTXASFLAGS)" -o hw4 hw4.cu sha256.cu

clean:
	rm -rf hw4 *.o
