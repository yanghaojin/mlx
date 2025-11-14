// Author: GreenBitAI 2025
// Optimized version with dynamic tile sizing for better batch parallelism

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/steel/mma.cuh"
#include "mlx/backend/cuda/steel/tiles.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// ============================================================================
// STAGE 1: Pointer Setup - SEGMENTED LAYOUT
// ============================================================================

__global__ void set_gather_qmm_pointers(
    void** pointers,
    const uint8_t* x_start,
    const uint8_t* w_start,
    const uint8_t* scales_start,
    const uint8_t* biases_start,
    uint8_t* y_start,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int x_itemsize,
    int scales_itemsize,
    int x_batch_stride,
    int w_batch_stride,
    int s_batch_stride,
    int b_batch_stride,
    int y_batch_stride,
    int batch_count) {

  int batch_idx = cg::this_grid().thread_rank();
  if (batch_idx >= batch_count) {
    return;
  }

  uint32_t x_idx = lhs_indices[batch_idx];
  uint32_t expert_idx = rhs_indices[batch_idx];

  pointers[batch_idx] =
      (void*)(x_start + x_idx * x_batch_stride * x_itemsize);
  pointers[batch_idx + batch_count] =
      (void*)(w_start + expert_idx * w_batch_stride);
  pointers[batch_idx + 2 * batch_count] =
      (void*)(scales_start + expert_idx * s_batch_stride * scales_itemsize);
  pointers[batch_idx + 3 * batch_count] =
      (void*)(biases_start + expert_idx * b_batch_stride * scales_itemsize);
  pointers[batch_idx + 4 * batch_count] =
      (void*)(y_start + batch_idx * y_batch_stride * x_itemsize);
}

// ============================================================================
// STAGE 2: Quantized Weight Loading
// ============================================================================

template <int NUM_WARPS, int group_size, int bits, typename T, typename Tile>
__device__ inline void load_quantized(
    Tile& tile,
    const uint8_t* x,
    const T* scales,
    const T* biases,
    int K,
    int valid_rows) {
  constexpr int NUM_THREADS = NUM_WARPS * 32;
  constexpr int ELEMENTS_PER_LOAD = sizeof(uint32_t) * get_pack_factor<bits>();
  constexpr int NUM_LOADS = Tile::NUMEL / ELEMENTS_PER_LOAD;
  constexpr int NUM_LOADS_PER_THREAD = NUM_LOADS / NUM_THREADS;
  constexpr int NUM_LOADS_PER_ROW = Tile::COLS / ELEMENTS_PER_LOAD;
  constexpr int STEP_ROWS = NUM_THREADS / NUM_LOADS_PER_ROW;
  constexpr int MASK = (1 << bits) - 1;

  const int row = threadIdx.x / NUM_LOADS_PER_ROW;
  const int col = threadIdx.x % NUM_LOADS_PER_ROW;

  const int Kx = K / get_pack_factor<bits>();
  const int Kg = K / group_size;

  x += row * Kx + col * (ELEMENTS_PER_LOAD / get_pack_factor<bits>());

  const T* scales_ptr = (row < valid_rows) ?
      (scales + row * Kg + col * ELEMENTS_PER_LOAD / group_size) : scales;
  const T* biases_ptr = (row < valid_rows) ?
      (biases + row * Kg + col * ELEMENTS_PER_LOAD / group_size) : biases;

  MLX_UNROLL
  for (int i = 0; i < NUM_LOADS_PER_THREAD; i++) {
    int current_row = row + i * STEP_ROWS;
    T vs[ELEMENTS_PER_LOAD];

    if (current_row < valid_rows) {
      uint32_t w = *reinterpret_cast<const uint32_t*>(x + i * STEP_ROWS * Kx);
      T s = scales_ptr[i * STEP_ROWS * Kg];
      T b = biases_ptr[i * STEP_ROWS * Kg];

      MLX_UNROLL
      for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
        T q_val = static_cast<T>((w >> (j * bits)) & MASK);
        vs[j] = q_val * s + b;
      }
    } else {
      MLX_UNROLL
      for (int j = 0; j < ELEMENTS_PER_LOAD; j++) {
        vs[j] = static_cast<T>(0);
      }
    }

    tile.store(vs, current_row, col * ELEMENTS_PER_LOAD);
  }
}

// ============================================================================
// STAGE 3: Batched QMM Kernel - OPTIMIZED WITH DYNAMIC TILE SIZING
// ============================================================================

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WARPS_M,
    int WARPS_N,
    int group_size,
    int bits,
    bool aligned_M>
__global__ void gather_qmm_batched_kernel(
    void** pointers,
    int M,
    int N,
    int K,
    int batch_count) {

  constexpr int NUM_WARPS = WARPS_M * WARPS_N;
  constexpr int WARP_STEP_M = BM / WARPS_M;
  constexpr int WARP_STEP_N = BN / WARPS_N;

  const int warpid = threadIdx.x / 32;
  const int laneid = threadIdx.x % 32;
  const int wm = warpid / WARPS_N;
  const int wn = warpid % WARPS_N;
  const int offset_m = wm * WARP_STEP_M;
  const int offset_n = wn * WARP_STEP_N;

  const int batch_idx = blockIdx.z;

  if (batch_idx >= batch_count) {
    return;
  }

  // Segmented layout reads
  const T* x_base = reinterpret_cast<const T*>(pointers[batch_idx]);
  const uint8_t* w_base = reinterpret_cast<const uint8_t*>(pointers[batch_idx + batch_count]);
  const T* scales_base = reinterpret_cast<const T*>(pointers[batch_idx + 2 * batch_count]);
  const T* biases_base = reinterpret_cast<const T*>(pointers[batch_idx + 3 * batch_count]);
  T* y_base = reinterpret_cast<T*>(pointers[batch_idx + 4 * batch_count]);

  extern __shared__ char shmem[];
  SharedTile<T, BM, BK>(&xs)[1] = *(SharedTile<T, BM, BK>(*)[1])(&shmem[0]);
  SharedTile<T, BN, BK>(&ws)[1] =
      *(SharedTile<T, BN, BK>(*)[1])(&shmem[1 * sizeof(T) * BM * BK]);

  RegisterTile<float, BM / WARPS_M, BN / WARPS_N> C;
  RegisterTile<T, BM / WARPS_M, 16> A;
  RegisterTile<T, BN / WARPS_N, 16> B;

  const int max_rows = M - blockIdx.y * BM;
  const int valid_weight_rows = min(BN, N - blockIdx.x * BN);

  const T* x_ptr = x_base + blockIdx.y * BM * K;
  const uint8_t* w_ptr = w_base + blockIdx.x * BN * K / get_pack_factor<bits>();
  const T* scales_ptr = scales_base + blockIdx.x * BN * K / group_size;
  const T* biases_ptr = biases_base + blockIdx.x * BN * K / group_size;
  T* y_ptr = y_base + blockIdx.y * BM * N + blockIdx.x * BN;

  C.fill(0);

  int tic = 0;
  uint32_t base_addr_xs[1], base_addr_ws[1];
  base_addr_xs[0] = __cvta_generic_to_shared(&xs[0].data[0]);
  base_addr_ws[0] = __cvta_generic_to_shared(&ws[0].data[0]);

  if (aligned_M || max_rows >= BM) {
    // Fast path
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async<NUM_WARPS>(xs[tic], base_addr_xs[tic], x_ptr + k_block, K);
      cp_async_commit();
      cp_async_wait_all();
      __syncthreads();

      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w_ptr + k_block / get_pack_factor<bits>(),
          scales_ptr + k_block / group_size,
          biases_ptr + k_block / group_size,
          K,
          valid_weight_rows);
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

    // Output with N-dimension boundary checking
    if (offset_n < N) {
      if (offset_n + WARP_STEP_N <= N) {
        C.store_global(y_ptr, N, offset_m, offset_n);
      } else {
        // Partial warp - element-wise store
        const int local_row_base = offset_m + (laneid / 4);
        const int local_col_base = offset_n + (laneid % 4) * 2;

        for (int tile_y = 0; tile_y < C.TILES_Y; tile_y++) {
          for (int tile_x = 0; tile_x < C.TILES_X; tile_x++) {
            auto& tile = C.data[tile_y * C.TILES_X + tile_x];
            int gr = local_row_base + tile_y * 16;
            int gc = local_col_base + tile_x * 16;

            if (gr < M && gc < N)
              y_ptr[gr * N + gc] = static_cast<T>(tile.values[0].x);
            if (gr < M && gc + 1 < N)
              y_ptr[gr * N + gc + 1] = static_cast<T>(tile.values[0].y);

            if (gr + 8 < M && gc < N)
              y_ptr[(gr + 8) * N + gc] = static_cast<T>(tile.values[1].x);
            if (gr + 8 < M && gc + 1 < N)
              y_ptr[(gr + 8) * N + gc + 1] = static_cast<T>(tile.values[1].y);

            if (gr < M && gc + 8 < N)
              y_ptr[gr * N + gc + 8] = static_cast<T>(tile.values[2].x);
            if (gr < M && gc + 9 < N)
              y_ptr[gr * N + gc + 9] = static_cast<T>(tile.values[2].y);

            if (gr + 8 < M && gc + 8 < N)
              y_ptr[(gr + 8) * N + gc + 8] = static_cast<T>(tile.values[3].x);
            if (gr + 8 < M && gc + 9 < N)
              y_ptr[(gr + 8) * N + gc + 9] = static_cast<T>(tile.values[3].y);
          }
        }
      }
    }
  } else {
    // Slow path with M boundary checking
    for (int k_block = 0; k_block < K; k_block += BK) {
      load_async_safe<NUM_WARPS>(
          xs[tic], base_addr_xs[tic], x_ptr + k_block, K, max_rows);
      cp_async_commit();
      cp_async_wait_all();
      __syncthreads();

      load_quantized<NUM_WARPS, group_size, bits>(
          ws[tic],
          w_ptr + k_block / get_pack_factor<bits>(),
          scales_ptr + k_block / group_size,
          biases_ptr + k_block / group_size,
          K,
          valid_weight_rows);
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

    // Output with both M and N bounds checking
    if (offset_n < N) {
      if (offset_n + WARP_STEP_N <= N) {
        C.store_global_safe(y_ptr, N, offset_m, offset_n, max_rows);
      } else {
        const int local_row_base = offset_m + (laneid / 4);
        const int local_col_base = offset_n + (laneid % 4) * 2;

        for (int tile_y = 0; tile_y < C.TILES_Y; tile_y++) {
          for (int tile_x = 0; tile_x < C.TILES_X; tile_x++) {
            auto& tile = C.data[tile_y * C.TILES_X + tile_x];
            int gr = local_row_base + tile_y * 16;
            int gc = local_col_base + tile_x * 16;

            if (gr < max_rows && gc < N)
              y_ptr[gr * N + gc] = static_cast<T>(tile.values[0].x);
            if (gr < max_rows && gc + 1 < N)
              y_ptr[gr * N + gc + 1] = static_cast<T>(tile.values[0].y);

            if (gr + 8 < max_rows && gc < N)
              y_ptr[(gr + 8) * N + gc] = static_cast<T>(tile.values[1].x);
            if (gr + 8 < max_rows && gc + 1 < N)
              y_ptr[(gr + 8) * N + gc + 1] = static_cast<T>(tile.values[1].y);

            if (gr < max_rows && gc + 8 < N)
              y_ptr[gr * N + gc + 8] = static_cast<T>(tile.values[2].x);
            if (gr < max_rows && gc + 9 < N)
              y_ptr[gr * N + gc + 9] = static_cast<T>(tile.values[2].y);

            if (gr + 8 < max_rows && gc + 8 < N)
              y_ptr[(gr + 8) * N + gc + 8] = static_cast<T>(tile.values[3].x);
            if (gr + 8 < max_rows && gc + 9 < N)
              y_ptr[(gr + 8) * N + gc + 9] = static_cast<T>(tile.values[3].y);
          }
        }
      }
    }
  }
}

} // namespace cu

// ============================================================================
// Host Function - WITH DYNAMIC TILE SIZE SELECTION
// ============================================================================

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
    throw std::invalid_argument(
        "[gather_qmm] Only bfloat16 and float16 are supported");
  }
  if (!transpose_) {
    throw std::invalid_argument(
        "[gather_qmm] Only transposed matmul is supported for now");
  }

  int B = rhs_indices.size();
  int x_batch_stride = M * K;
  int pack_factor = (bits_ == 3 || bits_ == 5) ? 8 : (bits_ == 6 ? 4 : 8 / bits_);
  int w_batch_stride = N * K / pack_factor;
  int s_batch_stride = N * K / group_size_;
  int b_batch_stride = N * K / group_size_;
  int y_batch_stride = M * N;

  // Allocate pointer array
  auto pointers = array(
      allocator::malloc(B * sizeof(void*) * 5),
      {B * 5},
      uint64);

  enc.add_temporary(pointers);

  // ========================================================================
  // STAGE 1: Setup pointers (unchanged)
  // ========================================================================

  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  enc.set_input_array(biases);
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(pointers);

  int pointer_setup_threads = std::min(B, 256);
  int pointer_setup_blocks = (B + pointer_setup_threads - 1) / pointer_setup_threads;

  enc.add_kernel_node(
      cu::set_gather_qmm_pointers,
      pointer_setup_blocks,
      pointer_setup_threads,
      0,
      pointers.data<void*>(),
      x.data<uint8_t>(),
      w.data<uint8_t>(),
      scales.data<uint8_t>(),
      biases.data<uint8_t>(),
      out.data<uint8_t>(),
      lhs_indices.data<uint32_t>(),
      rhs_indices.data<uint32_t>(),
      static_cast<int>(x.itemsize()),
      static_cast<int>(scales.itemsize()),
      x_batch_stride,
      w_batch_stride,
      s_batch_stride,
      b_batch_stride,
      y_batch_stride,
      B);

  // ========================================================================
  // STAGE 2: Batched quantized matmul - OPTIMIZED TILE SELECTION
  // ========================================================================

  enc.set_input_array(pointers);
  enc.set_input_array(x);
  enc.set_input_array(w);
  enc.set_input_array(scales);
  enc.set_input_array(biases);
  enc.set_input_array(lhs_indices);
  enc.set_input_array(rhs_indices);
  enc.set_output_array(out);

  dispatch_float_types(x.dtype(), "gather_qmm", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        constexpr int BK = 32;

        if (M <= 1 && B < 500) {
          // Chat scenario (Batch 1): Use original configuration
          // This is the only config that matches original 51ms performance
          constexpr int BM = 128;
          constexpr int BN = 128;
          constexpr int WARPS_M = 2;
          constexpr int WARPS_N = 4;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, false>;

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 2 && B < 900) {
          // Batch 2 scenario: Optimized config (best performance)
          constexpr int BM = 32;
          constexpr int BN = 128;
          constexpr int WARPS_M = 1;
          constexpr int WARPS_N = 4;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, false>;

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 8 && B < 1200) {
          // Batch 3-4 scenario: V1 config for best multi-batch performance
          constexpr int BM = 16;
          constexpr int BN = 64;
          constexpr int WARPS_M = 1;
          constexpr int WARPS_N = 2;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, false>;

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 8) {
          // Single-batch decode (Batch 1-2, M=2-8): Balance performance
          // Use original config for proven stability
          constexpr int BM = 128;
          constexpr int BN = 128;
          constexpr int WARPS_M = 2;
          constexpr int WARPS_N = 4;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, false>;

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 8) {
          // Multi-batch decode (Batch 3+): Optimize for throughput
          constexpr int BM = 16;
          constexpr int BN = 64;
          constexpr int WARPS_M = 1;
          constexpr int WARPS_N = 2;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, false>;

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 64) {
          // Small-medium batch decode: Standard 64x64 tile
          constexpr int BM = 64;
          constexpr int BN = 64;
          constexpr int WARPS_M = 2;
          constexpr int WARPS_N = 2;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, true>;

          if (M % BM != 0) {
            kernel = cu::gather_qmm_batched_kernel<
                DataType, BM, BN, BK, WARPS_M, WARPS_N,
                group_size.value, bits.value, false>;
          }

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else if (M <= 128) {
          // Medium batch: Increase BN for better throughput
          constexpr int BM = 64;
          constexpr int BN = 128;
          constexpr int WARPS_M = 2;
          constexpr int WARPS_N = 4;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, true>;

          if (M % BM != 0) {
            kernel = cu::gather_qmm_batched_kernel<
                DataType, BM, BN, BK, WARPS_M, WARPS_N,
                group_size.value, bits.value, false>;
          }

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);

        } else {
          // Large batch (Prefill): Maximum tile size
          constexpr int BM = 128;
          constexpr int BN = 128;
          constexpr int WARPS_M = 2;
          constexpr int WARPS_N = 4;

          auto kernel = cu::gather_qmm_batched_kernel<
              DataType, BM, BN, BK, WARPS_M, WARPS_N,
              group_size.value, bits.value, true>;

          if (M % BM != 0) {
            kernel = cu::gather_qmm_batched_kernel<
                DataType, BM, BN, BK, WARPS_M, WARPS_N,
                group_size.value, bits.value, false>;
          }

          dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, B);
          dim3 threads(WARPS_M * WARPS_N * 32, 1, 1);
          int shmem_size = sizeof(DataType) * (BM * BK + BN * BK);

          enc.add_kernel_node(
              kernel, grid, threads, shmem_size,
              pointers.data<void*>(), M, N, K, B);
        }
      });
    });
  });
}

} // namespace mlx::core