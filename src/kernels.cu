#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include "../tester/utils.h"

template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t rows,
                              size_t hidden_dim,
                              float eps) {
    // 动态共享内存，用于存储 warp 级别的局部和
    extern __shared__ float s_sum[];
    // 每个 block 负责一行，ty 表示行索引
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    size_t tid = threadIdx.x;
    size_t row_offset = row * hidden_dim;

    // Step 1: 每个线程计算自己负责元素的局部平方和
    T local_sum = 0.0;
    for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
        T val = static_cast<T>(input[row_offset + col]);
        local_sum += val * val;
    }

    // Step 2: Block 内归约求和（使用共享内存）
    s_sum[tid] = local_sum;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: 计算 RMS 缩放因子
    float mean_sq = s_sum[0] / static_cast<float>(hidden_dim);
    float rms = rsqrtf(mean_sq + eps);

    // Step 4: 每个线程重新遍历，计算并写回输出
    for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
        float val = static_cast<float>(input[row_offset + col]);
        float w = static_cast<float>(weight[col]);
        float out = val * rms * w;
        output[row_offset + col] = static_cast<T>(out);
    }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
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
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
    // TODO: Implement the rmsNorm function
    // 1. 声明变量与内存大小计算
    T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
    size_t input_bytes  = rows * hidden_dim * sizeof(T);
    size_t weight_bytes = hidden_dim * sizeof(T);
    // 2.内存的申请与分配(主机->设备)
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_weight, weight_bytes);
    cudaMalloc(&d_output, input_bytes);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice);
    // 3. 调用内核函数
    dim3 blockDim(256,1);
    dim3 gridDim(1,rows);

    size_t shared_mem_size = blockDim.x * sizeof(float);

    rmsNormKernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    // 4. 拷贝结果(设备->主机)
    cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost);

    // 5. 释放设备内存
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

template <typename T, int Br, int Bc>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    float softmax_scale)
{
    const int b  = blockIdx.y;          // batch index
    const int hq = blockIdx.x;          // query head index
    // GQA: map query head to kv head. Assumes query_heads % kv_heads == 0
    const int hk = hq * kv_heads / query_heads;
    const int tid = threadIdx.x;        // 0 .. Br-1
    const int q_tile_id = blockIdx.z;
    const int q_row = q_tile_id * Br + tid;

    // -------------------------------------------------------------------------
    // Shared Memory 布局
    // -------------------------------------------------------------------------
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    float* Q_tile = smem;                           // [Br][head_dim]
    float* K_tile = smem + Br * head_dim;           // [Bc][head_dim]
    float* V_tile = smem + (Br + Bc) * head_dim;    // [Bc][head_dim]

    // -------------------------------------------------------------------------
    // 寄存器状态
    // -------------------------------------------------------------------------
    float m_i = -FLT_MAX;   // running max
    float l_i = 0.0f;       // running sum of exp
    // 注意：由于head_dim是运行时参数，acc可能被编译器放入local memory。
    // 如果head_dim固定（如128），建议将head_dim也作为模板参数以使用寄存器。
    float acc[128];         

    // 初始化累加器
    // 必须确保 head_dim <= 128，否则这里会越界。Host端需要做检查。
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // Step 1: 加载 Q tile 到 Shared Memory
    // -------------------------------------------------------------------------
    if (q_row < tgt_seq_len) {
        const T* q_ptr = Q + ((b * tgt_seq_len + q_row) * query_heads + hq) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            Q_tile[tid * head_dim + d] = static_cast<float>(q_ptr[d]);
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: 遍历 K/V 的列方向 tile
    // -------------------------------------------------------------------------
    const int num_kv_tiles = (src_seq_len + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;

        // --- 优化：提前判断 Causal Mask ---
        // 如果整个 K/V tile 都在未来，直接跳过加载和计算，节省带宽
        if (is_causal && kv_start >= (q_tile_id + 1) * Br) {
            continue;
        }

        __syncthreads();

        // 协作加载 K_tile 和 V_tile
        const int total_kv = Bc * head_dim;
        for (int idx = tid; idx < total_kv; idx += Br) {
            int r = idx / head_dim;
            int d = idx % head_dim;
            int kv_row = kv_start + r;

            if (kv_row < src_seq_len) {
                const T* k_ptr = K + ((b * src_seq_len + kv_row) * kv_heads + hk) * head_dim;
                const T* v_ptr = V + ((b * src_seq_len + kv_row) * kv_heads + hk) * head_dim;
                K_tile[r * head_dim + d] = static_cast<float>(k_ptr[d]);
                V_tile[r * head_dim + d] = static_cast<float>(v_ptr[d]);
            } else {
                K_tile[r * head_dim + d] = 0.0f;
                V_tile[r * head_dim + d] = 0.0f;
            }
        }

        __syncthreads();

        if (q_row >= tgt_seq_len) continue;

        // ---------------------------------------------------------------------
        // Step 3: 计算 dot(Q_i, K_j) 并做 Online Softmax
        // ---------------------------------------------------------------------
        for (int j = 0; j < Bc; ++j) {
            int kv_row = kv_start + j;
            if (kv_row >= src_seq_len) break;

            // Causal mask：行级判断
            if (is_causal && q_row < kv_row) continue;

            float s_ij = 0.0f;
            // 循环展开有助于向量化
            #pragma unroll 4
            for (int d = 0; d < head_dim; ++d) {
                s_ij += Q_tile[tid * head_dim + d] * K_tile[j * head_dim + d];
            }
            s_ij *= softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_i, s_ij);
            // 修正数值稳定性：exp(x) 当 x过小会归零，过大不重要，无需特殊处理
            float exp_old = expf(m_i - m_new);
            float exp_new = expf(s_ij - m_new);

            #pragma unroll 4
            for (int d = 0; d < head_dim; ++d) {
                acc[d] = acc[d] * exp_old + exp_new * V_tile[j * head_dim + d];
            }

            l_i = l_i * exp_old + exp_new;
            m_i = m_new;
        }
    }
    
    // -------------------------------------------------------------------------
    // Step 4: 归一化并写回 Global Memory
    // -------------------------------------------------------------------------
    if (q_row < tgt_seq_len) {
        // 修正：防止 l_i 为 0 导致 NaN (例如全mask的情况)
        float norm = (l_i == 0.0f) ? 0.0f : (1.0f / l_i);
        
        T* o_ptr = O + ((b * tgt_seq_len + q_row) * query_heads + hq) * head_dim;
        
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            o_ptr[d] = static_cast<T>(acc[d] * norm);
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
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    // TODO: Implement the flash attention function
    // 1. 基础校验
    size_t q_elems = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_elems = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    if (h_q.size() != q_elems || h_k.size() != kv_elems || h_v.size() != kv_elems) {
        throw std::runtime_error("Input tensor size mismatch");
    }
    
    // 修正：限制 head_dim 以防止 kernel 内部的 acc 数组越界
    // 如果需要支持更大的 head_dim (如256)，需要修改 kernel 中的 acc[128] 大小
    if (head_dim > 128) {
        throw std::runtime_error("head_dim > 128 is not supported by this kernel implementation (acc buffer overflow risk).");
    }
    if (head_dim <= 0) {
        throw std::runtime_error("head_dim must be positive.");
    }

    h_o.resize(q_elems);

    // 2. 设备内存分配
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_bytes  = q_elems * sizeof(T);
    size_t kv_bytes = kv_elems * sizeof(T);

    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_o, q_bytes);

    cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    cudaStream_t stream = 0;
    cudaError_t err;

    // 3. Launch 配置
    // RTX 3050 具有 48KB default shared memory.
    // 使用 auto lambda 简化启动逻辑
    auto launch = [&](auto kernel, int Br, int Bc) {
        int num_q_tiles = (target_seq_len + Br - 1) / Br;
        if (num_q_tiles == 0) num_q_tiles = 1;

        dim3 grid(query_heads, batch_size, num_q_tiles);
        dim3 block(Br);

        if (grid.x > 65535 || grid.y > 65535 || grid.z > 65535) {
            throw std::runtime_error("Grid dimension exceeds CUDA limit");
        }

        // 计算共享内存大小 (float 类型)
        size_t smem_size = (Br + Bc + Bc) * head_dim * sizeof(float);
        
        // RTX 3050 (Ampere) 支持动态共享内存扩展
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        // 如果需要的 smem 超过默认值，尝试申请更多
        if (smem_size > prop.sharedMemPerBlock) {
             // 这里只是尝试设置，如果失败会由 kernel launch 返回错误
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        }

        kernel<<<grid, block, smem_size, stream>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal, softmax_scale
        );
    };

    // 根据维度选择 Tile 大小
    // head_dim=64 -> Br=64, smem=(64+64+64)*64*4 = 48KB (刚好卡在 3050 默认限制)
    // head_dim=128 -> Br=32, smem=(32+32+32)*128*4 = 48KB
    if (head_dim <= 64) {
        launch(flash_attention_kernel<T, 64, 64>, 64, 64);
    } else if (head_dim <= 128) {
        launch(flash_attention_kernel<T, 32, 32>, 32, 32);
    } else {
        // 理论上通过修改 acc 大小可以支持更大维度，但此处为了安全已在开头拦截
        launch(flash_attention_kernel<T, 16, 16>, 16, 16);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaMemcpy(h_o.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
