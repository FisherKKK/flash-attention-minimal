#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


// __global__ represents here is a cuda kernel, allowing codes execute in GPU issued by CPU
// Grid is the concept of the same parallel tasks
// A group of threads construct the block, block is for one task
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    // thread offset
    int tx = threadIdx.x;
    //TODO? why here are multi-dimension?  block offset
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    // gridDim(B, nh) like below, every grid correspond to attention calculation: 
    /**            ***********
     *             ***********
     *             ***********
     *             ***********
     *             ***********
     * bx * gridDim.y is batch offset and by * N * d is head offset
     */
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    // row offset + column offset
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q, K, V, S
    // extern __shared__ specify the SRAM
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram; // 0-offset
    float* Kj = &sram[tile_size]; // 1-offset
    float* Vj = &sram[tile_size * 2]; // 2-offset
    float* S = &sram[tile_size * 3]; // 3-offset

    // Outer loop of KV
    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        // For specific thread, load sram from hram
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        // Inner loop of Q
        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            // Record maximum
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            // Record sum
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                // Matrix Multiple
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            // Calculate score with scale
            // All of the score normalized by the maximum
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                // Record the sum
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m(maximum) and l(sum(exp))
            float row_m_new = max(row_m_prev, row_m);
            // PreSum * exp(PreM - MaxM) + CurSum * exp(CurM - MaxM)
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                // O = V * score, but we should consider about the previous calculation result
                // First O needs recovery score and then continune calculation --> Second line
                // Divide new score sum --> First line
                // Sum new internal result --> Third line
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            // Update result
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

// Q(B, H, T, C), same with the K, V
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    // Here is to say the block size
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    // Tc --> Divide K, V
    // Tr --> Dividie Q
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    // First item: K(Bc, d), V(Bc, d), O(Br, d), l(Br), m(Br)
    // Second item: S(Br, Bc, d), intermediate result

    //TODO?: Calculation may be mistaken
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    // Get size of max SRAM
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    // This is the parallel model for computation
    // Q, K, V are full original version
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}