
#pragma once

#include "dCSR_sp.h"
#include "Timings.h"
#include "spECKConfig.h"

// REPLACE THESE VALUES WITH YOUR ACTUAL DEVICE SPECIFICATIONS

static constexpr int spECK_STATIC_MEM_PER_BLOCK {49152};
static constexpr int spECK_DYNAMIC_MEM_PER_BLOCK{98304};

namespace spECK
{
    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyspECK(const spECKWrapper::dCSR<DataType> &A, const spECKWrapper::dCSR<DataType> &B, spECKWrapper::dCSR<DataType> &matOut, spECKConfig &config, Timings &timings);

    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyspECK(unsigned int* A_row_offsets, unsigned int* A_col_ids, DataType* A_data,
                        int A_rows, int A_cols, int A_nnz,  
                        unsigned int * B_row_offsets, unsigned int* B_col_ids, DataType* B_data,
                        int B_rows, int B_cols, int B_nnz,  
                        spECKWrapper::dCSR<DataType> &matOut, spECKConfig &config, Timings &timings);

} // namespace spECK
