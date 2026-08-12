#include <vector>
#include <cstdio>
#include <stdexcept>
#include <cmath>
#include <cfloat>
#include <musa_fp16.h>
#include "../tester/utils.h"

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 * output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input, const T* __restrict__ weight, T* __restrict__ output, size_t rows, size_t hidden_dim, float eps) {
    extern __shared__ float s_sum[];
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    size_t tid = threadIdx.x;
    size_t row_offset = row * hidden_dim;
    float local_sum = 0.0f;
    for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
        float val = static_cast<float>(input[row_offset + col]);
        local_sum += val * val;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_sum[0] / static_cast<float>(hidden_dim);
    float rms = rsqrtf(mean_sq + eps);

    for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
        float val = static_cast<float>(input[row_offset + col]);
        float w = static_cast<float>(weight[col]);
        float out = val * rms * w;
        output[row_offset + col] = static_cast<T>(out);
    }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight, std::vector<T>& h_output, size_t rows, size_t hidden_dim, float eps) {
    T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    size_t input_bytes = rows * hidden_dim * sizeof(T);
    size_t weight_bytes = hidden_dim * sizeof(T);

    musaMalloc(&d_input, input_bytes);
    musaMalloc(&d_weight, weight_bytes);
    musaMalloc(&d_output, input_bytes);

    musaMemcpy(d_input, h_input.data(), input_bytes, musaMemcpyHostToDevice);
    musaMemcpy(d_weight, h_weight.data(), weight_bytes, musaMemcpyHostToDevice);

    dim3 blockDim(256, 1);
    dim3 gridDim(1, rows);
    size_t shared_mem_size = blockDim.x * sizeof(float);

    rmsNormKernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps);

    musaDeviceSynchronize();
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        fprintf(stderr, "MUSA kernel failed: %s\n", musaGetErrorString(err));
    }

    musaMemcpy(h_output.data(), d_output, input_bytes, musaMemcpyDeviceToHost);

    musaFree(d_input);
    musaFree(d_weight);
    musaFree(d_output);
}

template <typename T, int BLOCK_SIZE_Q, int BLOCK_SIZE_KV>
__global__ void flash_attention_kernel_original_softmax(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size, int target_seq_len, int source_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal,
    float softmax_scale) {

    const int batch_idx = blockIdx.y;
    const int query_head_idx = blockIdx.x;
    const int kv_head_idx = query_head_idx * kv_heads / query_heads;
    const int thread_idx = threadIdx.x;
    const int query_tile_idx = blockIdx.z;
    const int global_query_row_idx = query_tile_idx * BLOCK_SIZE_Q + thread_idx;

    extern __shared__ char shared_mem_buffer[];
    float* shared_mem_ptr = reinterpret_cast<float*>(shared_mem_buffer);
    float* s_query = shared_mem_ptr;
    float* s_key = shared_mem_ptr + BLOCK_SIZE_Q * head_dim;
    float* s_value = shared_mem_ptr + (BLOCK_SIZE_Q + BLOCK_SIZE_KV) * head_dim;

    float accumulated_output[128];
    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        accumulated_output[dim_idx] = 0.0f;
    }

    if (global_query_row_idx < target_seq_len) {
        const T* query_ptr = Q + ((batch_idx * target_seq_len + global_query_row_idx) * query_heads + query_head_idx) * head_dim;
        for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            s_query[thread_idx * head_dim + dim_idx] = static_cast<float>(query_ptr[dim_idx]);
        }
    }

    const int num_kv_blocks = (source_seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;
    float current_max_val = -FLT_MAX;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_block_offset = kv_block_idx * BLOCK_SIZE_KV;
        __syncthreads();

        const int elements_per_block = BLOCK_SIZE_KV * head_dim;
        for (int element_idx = thread_idx; element_idx < elements_per_block; element_idx += BLOCK_SIZE_Q) {
            int row_local = element_idx / head_dim;
            int dim_local = element_idx % head_dim;
            int global_kv_row_idx = kv_block_offset + row_local;
            if (global_kv_row_idx < source_seq_len) {
                const T* key_ptr = K + ((batch_idx * source_seq_len + global_kv_row_idx) * kv_heads + kv_head_idx) * head_dim;
                s_key[row_local * head_dim + dim_local] = static_cast<float>(key_ptr[dim_local]);
            } else {
                s_key[row_local * head_dim + dim_local] = 0.0f;
            }
        }
        __syncthreads();

        if (global_query_row_idx < target_seq_len) {
            for (int kv_idx_in_block = 0; kv_idx_in_block < BLOCK_SIZE_KV; ++kv_idx_in_block) {
                int global_kv_row_idx = kv_block_offset + kv_idx_in_block;
                if (global_kv_row_idx >= source_seq_len) break;
                if (is_causal && global_query_row_idx < global_kv_row_idx) continue;

                float attention_score = 0.0f;
                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    attention_score += s_query[thread_idx * head_dim + dim_idx] * s_key[kv_idx_in_block * head_dim + dim_idx];
                }
                attention_score *= softmax_scale;

                if (attention_score > current_max_val) {
                    current_max_val = attention_score;
                }
            }
        }
    }

    float denominator_sum = 0.0f;
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_block_offset = kv_block_idx * BLOCK_SIZE_KV;
        __syncthreads();

        const int elements_per_block = BLOCK_SIZE_KV * head_dim;
        for (int element_idx = thread_idx; element_idx < elements_per_block; element_idx += BLOCK_SIZE_Q) {
            int row_local = element_idx / head_dim;
            int dim_local = element_idx % head_dim;
            int global_kv_row_idx = kv_block_offset + row_local;
            if (global_kv_row_idx < source_seq_len) {
                const T* key_ptr = K + ((batch_idx * source_seq_len + global_kv_row_idx) * kv_heads + kv_head_idx) * head_dim;
                const T* value_ptr = V + ((batch_idx * source_seq_len + global_kv_row_idx) * kv_heads + kv_head_idx) * head_dim;
                s_key[row_local * head_dim + dim_local] = static_cast<float>(key_ptr[dim_local]);
                s_value[row_local * head_dim + dim_local] = static_cast<float>(value_ptr[dim_local]);
            } else {
                s_key[row_local * head_dim + dim_local] = 0.0f;
                s_value[row_local * head_dim + dim_local] = 0.0f;
            }
        }
        __syncthreads();

        if (global_query_row_idx < target_seq_len) {
            for (int kv_idx_in_block = 0; kv_idx_in_block < BLOCK_SIZE_KV; ++kv_idx_in_block) {
                int global_kv_row_idx = kv_block_offset + kv_idx_in_block;
                if (global_kv_row_idx >= source_seq_len) break;
                if (is_causal && global_query_row_idx < global_kv_row_idx) continue;

                float attention_score = 0.0f;
                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    attention_score += s_query[thread_idx * head_dim + dim_idx] * s_key[kv_idx_in_block * head_dim + dim_idx];
                }
                attention_score *= softmax_scale;

                float prob_weight = expf(attention_score - current_max_val);
                denominator_sum += prob_weight;
                for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                    accumulated_output[dim_idx] += prob_weight * s_value[kv_idx_in_block * head_dim + dim_idx];
                }
            }
        }
    }

    // 归一化并写回
    if (global_query_row_idx < target_seq_len) {
        float norm_factor = (denominator_sum == 0.0f) ? 0.0f : (1.0f / denominator_sum);
        T* output_ptr = O + ((batch_idx * target_seq_len + global_query_row_idx) * query_heads + query_head_idx) * head_dim;
        for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            output_ptr[dim_idx] = static_cast<T>(accumulated_output[dim_idx] * norm_factor);
        }
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k, const std::vector<T>& h_v, std::vector<T>& h_o, int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal){
    size_t num_query_elements = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t num_kv_elements = (size_t)batch_size * src_seq_len * kv_heads * head_dim;

    if (h_q.size() != num_query_elements || h_k.size() != num_kv_elements || h_v.size() != num_kv_elements) {
        throw std::runtime_error("Input tensor size mismatch");
    }
    if (head_dim > 128) {
        throw std::runtime_error("head_dim > 128 is not supported by this kernel implementation.");
    }
    if (head_dim <= 0) {
        throw std::runtime_error("head_dim must be positive.");
    }
    if (query_heads % kv_heads != 0) {
        throw std::runtime_error("query_heads must be divisible by kv_heads for GQA.");
    }

    h_o.resize(num_query_elements);
    T *device_q, *device_k, *device_v, *device_o;
    size_t query_bytes = num_query_elements * sizeof(T);
    size_t kv_bytes = num_kv_elements * sizeof(T);

    musaMalloc(&device_q, query_bytes);
    musaMalloc(&device_k, kv_bytes);
    musaMalloc(&device_v, kv_bytes);
    musaMalloc(&device_o, query_bytes);

    musaMemcpy(device_q, h_q.data(), query_bytes, musaMemcpyHostToDevice);
    musaMemcpy(device_k, h_k.data(), kv_bytes, musaMemcpyHostToDevice);
    musaMemcpy(device_v, h_v.data(), kv_bytes, musaMemcpyHostToDevice);

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    musaStream_t stream = 0;
    musaError_t musa_error;

    auto launch_kernel = [&](auto kernel, int block_size_q, int block_size_kv) {
        int num_query_blocks = (target_seq_len + block_size_q - 1) / block_size_q;
        if (num_query_blocks == 0) num_query_blocks = 1;
        dim3 grid(query_heads, batch_size, num_query_blocks);
        dim3 block(block_size_q);

        if (grid.x > 65535 || grid.y > 65535 || grid.z > 65535) {
            throw std::runtime_error("Grid dimension exceeds MUSA limit");
        }

        size_t dynamic_shared_mem_size = (block_size_q + block_size_kv + block_size_kv) * head_dim * sizeof(float);
        const void* kernel_ptr = reinterpret_cast<const void*>(kernel);
        
        musaError_t attribute_error = musaFuncSetAttribute(
            kernel_ptr,
            musaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dynamic_shared_mem_size)
        );

        kernel<<<grid, block, dynamic_shared_mem_size, stream>>>(
            device_q, device_k, device_v, device_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim, is_causal, softmax_scale
        );
    };

    if (head_dim <= 64) {
        launch_kernel(flash_attention_kernel_original_softmax<T, 64, 64>, 64, 64);
    } else if (head_dim <= 128) {
        launch_kernel(flash_attention_kernel_original_softmax<T, 32, 32>, 32, 32);
    } else {
        launch_kernel(flash_attention_kernel_original_softmax<T, 16, 16>, 16, 16);
    }

    musa_error = musaGetLastError();
    if (musa_error != musaSuccess) {
        musaFree(device_q);
        musaFree(device_k);
        musaFree(device_v);
        musaFree(device_o);
        throw std::runtime_error(std::string("Kernel launch failed: ") + musaGetErrorString(musa_error));
    }

    musaMemcpy(h_o.data(), device_o, query_bytes, musaMemcpyDeviceToHost);

    musaFree(device_q);
    musaFree(device_k);
    musaFree(device_v);
    musaFree(device_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&, std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&, std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, std::vector<float>&, int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&, const std::vector<half>&, std::vector<half>&, int, int, int, int, int, int, bool);
