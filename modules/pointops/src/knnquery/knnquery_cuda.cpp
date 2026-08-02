#include <vector>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knnquery_cuda_kernel.h"


void knnquery_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    TORCH_CHECK(
        nsample >= 1 && nsample <= KNNQUERY_MAX_NEIGHBORS,
        "knnquery nsample must be between 1 and ",
        KNNQUERY_MAX_NEIGHBORS,
        "; got ",
        nsample
    );
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    knnquery_cuda_launcher(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}
