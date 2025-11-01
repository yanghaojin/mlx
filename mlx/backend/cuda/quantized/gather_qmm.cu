// Author: GreenBitAI 2025
// Gather QMM - Step 1: Basic gather support for quantized matmul

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/matmul/mma.cuh"
#include "mlx/backend/cuda/matmul/tiles.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace cu {

// GPTQ-style quantization: groups along K dimension, not N
// Formula: dequant_val = q * scale + bias (where bias = -zero from GPTQ)
template <int NUM_WARPS, int group_size, int bits, typename T, typename Tile>
__device__ inline void load_quantized(
    Tile& tile,
    const uint8_t* x,
    const T* scales,
    const T* biases,
    int N,
    int valid_rows) {  // Number of valid output channels
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(uint32_t) * get_pack_factor<bits>();
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;
  constexpr int MASK = (1 << bits) - 1;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  const int Nx = N / get_pack_factor<bits>();
  const int Ng = N / group_size;

  x += row * Nx + col * (ELEMENTS_PER_LOAD / get_pack_factor<bits>());

  // Only add offset if within valid range, otherwise point to first row (will be ignored)
  const T* scales_ptr = (row < valid_rows) ?
      (scales + row * Ng + col * ELEMENTS_PER_LOAD / group_size) : scales;
  const T* biases_ptr = (row < valid_rows) ?
      (biases + row * Ng + col * ELEMENTS_PER_LOAD / group_size) : biases;

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    int current_row = row + i * STEP_ROWS;
    T vs[ELEMENTS_PER_LOAD];

    if (current_row < valid_rows) {
      // Process valid rows normally
      uint32_t w = *reinterpret_cast<const uint32_t*>(x + i * STEP_ROWS * Nx);
      T s = scales_ptr[i * STEP_ROWS * Ng];
      T b = biases_ptr[i * STEP_ROWS * Ng];

      MLX_UNROLL
      for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
        T q_val = static_cast<T>((w >> (j * bits)) & MASK);
        vs[j] = q_val * s + b;
      }
    } else {
      // Fill invalid rows with zeros
      MLX_UNROLL
      for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
        vs[j] = static_cast<T>(0);
      }
    }

    tile.store(vs, current_row, col * ELEMENTS_PER_LOAD);
  }
}

// Simple gather version - each block processes one batch element
// using indices to select which x and w matrices to use
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int group_size,
    int bits,
    bool aligned_M>
__global__ void gather_qmm_t_simple(
    const T* x,
    const uint8_t* w,
    const T* scales,
    const T* biases,
    const uint32_t* lhs_indices,  // indices for x selection [B]
    const uint32_t* rhs_indices,  // indices for w selection [B]
    T* y,
    int M,
    int N,
    int K,
    int x_batch_stride,  // stride between batches in x
    int w_batch_stride,  // stride between batches in w
    int s_batch_stride,  // stride between batches in scales
    int b_batch_stride)  // stride between batches in biases
{
  constexpr int WARPS_M = 2;
  constexpr int WARPS_N = 4;
  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int WARP_STEP_M = BM / WARPS_M;
  constexpr int WARP_STEP_N = BN / WARPS_N;

  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WARPS_N;
  const int wn = warpid % WARPS_N;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;

  // Get batch index for this block
  const int batch_idx = blockIdx.z;

  // Read indices to determine which matrices to use
  const uint32_t x_idx = lhs_indices[batch_idx];
  const uint32_t w_idx = rhs_indices[batch_idx];

  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&xs)[1] = *(SharedTile<T, BM, BK>(*)[1])(&shmem[0]);
  SharedTile<T, BN, BK>(&ws)[1] =
      *(SharedTile<T, BN, BK>(*)[1])(&shmem[1 * sizeof(T) * BM * BK]);

  RegisterTile<float, BM / WARPS_M, BN / WARPS_N> C;
  RegisterTile<T, BM / WARPS_M, 16> A;
  RegisterTile<T, BN / WARPS_N, 16> B;

  const int max_rows = M - blockIdx.y * BM;
  const int valid_weight_rows = min(BN, N - blockIdx.x * BN);

  // Use indices to select correct matrices
  const T* x_ptr = x + x_idx * x_batch_stride + blockIdx.y * BM * K;
  const uint8_t* w_ptr = w + w_idx * w_batch_stride + blockIdx.x * BN * K / get_pack_factor<bits>();
  const T* scales_ptr = scales + w_idx * s_batch_stride + blockIdx.x * BN * K / group_size;
  const T* biases_ptr = biases + w_idx * b_batch_stride + blockIdx.x * BN * K / group_size;
  T* y_ptr = y + batch_idx * M * N + blockIdx.y * BM * N + blockIdx.x * BN;

  C.fill(0);

  int tic = 0;
  uint32_t base_addr_xs[1], base_addr_ws[1];
  base_addr_xs[0] = __cvta_generic_to_shared(&xs[0].data[0]);
  base_addr_ws[0] = __cvta_generic_to_shared(&ws[0].data[0]);

  if (aligned_M || max_rows >= BM) {
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async<NUM_WARPS>(xs[tic], base_addr_xs[tic], x_ptr + k_block, K);
      cp_async_commit();
      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w_ptr + k_block / get_pack_factor<bits>(),
          scales_ptr + k_block / group_size,
          biases_ptr + k_block / group_size,
          K,
          valid_weight_rows);
      cp_async_wait_all();
      __syncthreads();

      MLX_UNROLL
      for (int k = 0; k < BK / 16; k++) {
        A.load(
            xs[tic],
            base_addr_xs[tic],
            offset_m + laneid % 16,
            k * 16 + laneid / 16 * 8);
        B.load(
            ws[tic],
            base_addr_ws[tic],
            offset_n + laneid % 16,
            k * 16 + laneid / 16 * 8);
        mma_t(C, A, B);
      }
    }

    if (offset_n < N) {
      C.store_global(y_ptr, N, offset_m, offset_n);
      __threadfence();
    }
  } else {
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async_safe<NUM_WARPS>(
          xs[tic], base_addr_xs[tic], x_ptr + k_block, K, max_rows);
      cp_async_commit();
      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w_ptr + k_block / get_pack_factor<bits>(),
          scales_ptr + k_block / group_size,
          biases_ptr + k_block / group_size,
          K,
          valid_weight_rows);
      cp_async_wait_all();
      __syncthreads();

      MLX_UNROLL
      for (int k = 0; k < BK / 16; k++) {
        A.load(
            xs[tic],
            base_addr_xs[tic],
            offset_m + laneid % 16,
            k * 16 + laneid / 16 * 8);
        B.load(
            ws[tic],
            base_addr_ws[tic],
            offset_n + laneid % 16,
            k * 16 + laneid / 16 * 8);
        mma_t(C, A, B);
      }
    }

    if (offset_n < N) {
		C.store_global_safe(y_ptr, N, offset_m, offset_n, max_rows);
		__threadfence();
    }
  }
}

} // namespace cu

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose_,
    int group_size_,
    int bits_,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc,
    const Stream& s) {

  if (x.dtype() != bfloat16 && x.dtype() != float16) {
    throw std::invalid_argument("[gather_qmm] Only bfloat16 and float16 are supported");
  }
  if (!transpose_) {
    throw std::invalid_argument(
        "[gather_qmm] Only transposed matmul is supported for now");
  }

  int B = out.size() / M / N;  // number of batches

  // Calculate batch strides carefully
  // x: (B, M, K) in elements of type T
  int x_batch_stride = M * K;  // elements

  // w: (B, N, K_packed) uint32, treated as uint8_t*
  // For uint32 storage: each element is 4 bytes
  // But MLX stores as uint8_t internally for packing
  // Need to calculate actual byte stride
  int pack_factor = (bits_ == 3 || bits_ == 5) ? 8 : (bits_ == 6 ? 4 : 8 / bits_);

  // Each row has K values, packed into K/pack_factor bytes
  // For N rows: N * (K / pack_factor) bytes per batch
  // But w.shape[-1] is already in packed units (K/pack_factor)
  // So actual stride is: N * w.shape(-1) * element_size
  int w_packed_elements_per_batch = N * (K / pack_factor);

  // For 4-bit: pack_factor = 2, so 2 values per byte
  // K=128 values → 64 bytes per row
  // N=2 rows → 128 bytes per batch
  int w_batch_stride = N * K / pack_factor;  // This should be bytes for 4-bit

  // Correct calculation based on actual storage:
  // w.shape() should be (B, N, K_packed) where K_packed depends on dtype
  // If dtype is uint32: each stores multiple packed values
  // If dtype is uint8: each stores values based on bits
  int w_actual_stride = w.shape(-2) * w.shape(-1) * size_of(w.dtype());

  int s_batch_stride = N * K / group_size_;
  int b_batch_stride = N * K / group_size_;

  // Register arrays
  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  enc.set_input_array(biases);
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);

  cudaMemsetAsync(out.data<void>(), 0, out.nbytes(), enc.stream());

  dispatch_float_types(x.dtype(), "gather_qmm", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int BK = 32;

        auto kernel = cu::gather_qmm_t_simple<
            DataType, BM, BN, BK,
            group_size.value, bits.value, true>;

        if (M % BM != 0) {
          kernel = cu::gather_qmm_t_simple<
              DataType, BM, BN, BK,
              group_size.value, bits.value, false>;
        }

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);

        enc.add_kernel_node(
            kernel,
            grid,
            dim3(2 * 4 * 32, 1, 1),
            1 * sizeof(DataType) * (BM * BK + BN * BK),
            x.data<DataType>(),
            w.data<uint8_t>(),
            scales.data<DataType>(),
            biases.data<DataType>(),
            lhs_indices.data<uint32_t>(),
            rhs_indices.data<uint32_t>(),
            out.data<DataType>(),
            M,
            N,
            K,
            x_batch_stride,
            w_batch_stride,
            s_batch_stride,
            b_batch_stride);
      });
    });
  });
}

} // namespace mlx::core