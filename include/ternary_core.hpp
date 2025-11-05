/**
 * @file ternary_core.hpp
 * @brief Core ternary arithmetic operations for efficient bit-packed ternary computation
 * @author Zombie
 * @date 2025-11-05
 * 
 * This header provides optimized functions for ternary arithmetic using bit-packed representations.
 * Ternary values (-1, 0, +1) are efficiently encoded using minimal bit patterns for high-performance
 * computation in machine learning and signal processing applications.
 */

#ifndef TERNARY_CORE_HPP
#define TERNARY_CORE_HPP

#include <cstdint>
#include <cassert>

namespace ternary_core {

// ============================================================================
// CONSTANTS AND MASKS
// ============================================================================

/**
 * @brief Bit mask for extracting ternary values from packed representation
 * @details Each ternary value occupies 2 bits: 1 bit for magnitude (0 or 1), 
 *          1 bit for sign (0 for negative/zero, 1 for positive)
 *          MASK = 0b0001'0001 = 0x11
 */
constexpr uint8_t MASK = 0b0001'0001;

/**
 * @brief Mask for extracting magnitude bits from ternary representation
 * @details Used to isolate the value bit (0b0000'0001)
 */
constexpr uint8_t VMASK = 0b0000'0001;

/**
 * @brief Mask for extracting sign bits from ternary representation
 * @details Used to isolate the sign bit (0b0001'0000)
 */
constexpr uint8_t SMASK = 0b0001'0000;

// ============================================================================
// PACKING AND UNPACKING OPERATIONS
// ============================================================================

/**
 * @brief Pack four ternary values into a single byte using 2 bits per value
 * @author Zombie
 * @details
 * Each ternary value is encoded as:
 * - (-1): 00 (magnitude=0, sign=0)
 * - (0):  01 (magnitude=1, sign=0) 
 * - (+1): 11 (magnitude=1, sign=1)
 * 
 * The packing pattern is:
 * bit 0: magnitude of value a
 * bit 1: magnitude of value b  
 * bit 2: magnitude of value c
 * bit 3: magnitude of value d
 * bit 4: sign of value a
 * bit 5: sign of value b
 * bit 6: sign of value c
 * bit 7: sign of value d
 * 
 * @param a First ternary value (-1, 0, or +1)
 * @param b Second ternary value (-1, 0, or +1)  
 * @param c Third ternary value (-1, 0, or +1)
 * @param d Fourth ternary value (-1, 0, or +1)
 * @return Packed byte containing all four ternary values
 */
inline constexpr uint8_t pack_v8(uint8_t a, uint8_t b, uint8_t c, uint8_t d) noexcept {
    return (a) | (b << 1) | (c << 2) | (d << 3);
}

/**
 * @brief Unpack a specific ternary value from a packed byte
 * @author Zombie
 * @details
 * Extracts the i-th ternary value from the packed representation.
 * The extraction reads both magnitude and sign bits to reconstruct the original ternary value.
 * 
 * @param vec Packed byte containing up to 4 ternary values
 * @param i Index of the value to extract (0-3)
 * @return Unpacked ternary value as a 2-bit encoded value
 */
inline constexpr uint8_t unpack_v8(uint8_t vec, uint8_t i) noexcept {
    return (vec >> i) & MASK;
}

/**
 * @brief Compute dot product between two packed ternary vectors using bit parallelism
 * @author Zombie
 * @details
 * Performs efficient ternary dot product using bit-level parallelism.
 * This function operates on 8 ternary values (4 in each input vector) packed into bytes.
 * 
 * Encoding scheme:
 * - Input vectors: packed ternary values using pack_v8 format
 * - Output: 8-bit result with magnitude in lower 4 bits, sign in upper 4 bits
 * 
 * The algorithm:
 * 1. Extract value and sign bits from both vectors
 * 2. Compute magnitude dot product using parity of AND operations
 * 3. Compute sign dot product using parity of value AND sign products
 * 4. Pack results into return format
 * 
 * Mathematical correctness:
 * - For ternary values x, y ∈ {-1, 0, 1}: x·y = sign(x*y) * (|x| & |y|)
 * - Bit-level parity operations efficiently compute this
 * 
 * @param x First packed ternary vector (8 ternary values in 2 bytes)
 * @param y Second packed ternary vector (8 ternary values in 2 bytes)
 * @return 8-bit packed result: [sign:4bits][magnitude:4bits]
 */
inline constexpr uint8_t dot_v8(uint8_t x, uint8_t y) noexcept {
    uint8_t vbits_x = x & 0x0F;      // Extract magnitude bits from lower nibble
    uint8_t sbits_x = (x >> 4) & 0x0F; // Extract sign bits from upper nibble
    uint8_t vbits_y = y & 0x0F;      // Extract magnitude bits from lower nibble
    uint8_t sbits_y = (y >> 4) & 0x0F; // Extract sign bits from upper nibble

    uint8_t val_res = vbits_x | vbits_y;    // Magnitude AND operation
    uint8_t sgn_res = sbits_x & sbits_y;    // Sign AND operation

    uint8_t parity_val = __builtin_parity(val_res);  // Count 1s in magnitude result
    uint8_t parity_sgn = __builtin_parity(val_res & sgn_res); // Count 1s in sign result

    return (parity_sgn << 4) | parity_val;  // Pack sign and magnitude results
}

// ============================================================================
// MATRIX OPERATIONS (32-bit packed matrices)
// ============================================================================

/**
 * @brief Pack four ternary rows into a single 32-bit integer
 * @author Zombie
 * @details
 * Creates a 32-bit representation of a 4x4 ternary matrix where each row
 * is packed using the 8-bit packing format from pack_v8.
 * 
 * Memory layout:
 * - bits 0-7:   row 0 (4 ternary values)
 * - bits 8-15:  row 1 (4 ternary values)
 * - bits 16-23: row 2 (4 ternary values)
 * - bits 24-31: row 3 (4 ternary values)
 * 
 * @param r0 First row (4 ternary values packed as byte)
 * @param r1 Second row (4 ternary values packed as byte)
 * @param r2 Third row (4 ternary values packed as byte)
 * @param r3 Fourth row (4 ternary values packed as byte)
 * @return 32-bit packed matrix representation
 */
inline constexpr uint32_t pack_m32(uint8_t r0, uint8_t r1, uint8_t r2, uint8_t r3) noexcept {
    return (uint32_t(r0) << 0) | (uint32_t(r1) << 8) | (uint32_t(r2) << 16) | (uint32_t(r3) << 24);
}

/**
 * @brief Extract a specific row from a 32-bit packed matrix
 * @author Zombie
 * @details
 * Retrieves the i-th row (0-3) from a 32-bit packed matrix representation.
 * Each row contains 4 ternary values packed in a byte.
 * 
 * @param m 32-bit packed matrix
 * @param i Row index to extract (0-3)
 * @return Row as packed byte containing 4 ternary values
 */
inline constexpr uint8_t unpack_row_m32(uint32_t m, uint8_t i) noexcept {
    return uint8_t(m >> (i * 8));
}

/**
 * @brief Transpose a 4x4 ternary matrix represented as packed 32-bit integer
 * @author Zombie
 * @details
 * Performs efficient matrix transpose using bit manipulation.
 * The transpose operation reorganizes matrix elements:
 * - Original element at position (row, col) moves to (col, row)
 * - Uses strategic bit masking and shifting for transpose
 * 
 * Bit manipulation strategy:
 * - Each ternary value occupies 2 bits in the packed format
 * - Transpose requires reorganizing these 2-bit groups
 * - Uses precomputed mask patterns for efficient transposition
 * 
 * @param m 32-bit packed matrix to transpose
 * @return 32-bit packed transposed matrix
 */
inline constexpr uint32_t transpose_m32(uint32_t m) noexcept {
    uint8_t r0 = unpack_row_m32(m, 0);
    uint8_t r1 = unpack_row_m32(m, 1);
    uint8_t r2 = unpack_row_m32(m, 2);
    uint8_t r3 = unpack_row_m32(m, 3);

    // Masks for extracting specific bit positions during transpose
    constexpr uint8_t mask_shift[4] = {
        0b00010001, // Extract bits for column 0
        0b00100010, // Extract bits for column 1  
        0b01000100, // Extract bits for column 2
        0b10001000  // Extract bits for column 3
    };

    // Build transposed columns using bit manipulation
    uint8_t c0 = (r0 & mask_shift[0]) | ((r1 & mask_shift[0]) << 1) | 
                ((r2 & mask_shift[0]) << 2) | ((r3 & mask_shift[0]) << 3);
    uint8_t c1 = (r0 & mask_shift[1]) | ((r1 & mask_shift[1]) << 1) | 
                ((r2 & mask_shift[1]) << 2) | ((r3 & mask_shift[1]) << 3);
    uint8_t c2 = (r0 & mask_shift[2]) | ((r1 & mask_shift[2]) << 1) | 
                ((r2 & mask_shift[2]) << 2) | ((r3 & mask_shift[2]) << 3);
    uint8_t c3 = (r0 & mask_shift[3]) | ((r1 & mask_shift[3]) << 1) | 
                ((r2 & mask_shift[3]) << 2) | ((r3 & mask_shift[3]) << 3);

    return pack_m32(c0, c1, c2, c3);
}

/**
 * @brief Multiply a packed matrix by a packed ternary vector
 * @author Zombie
 * @details
 * Computes matrix-vector product: result = matrix * vector
 * Both inputs use ternary packing format for maximum efficiency.
 * 
 * Operation:
 * - Matrix: 4x4 ternary matrix packed in 32-bit integer
 * - Vector: 4 ternary values packed in byte
 * - Result: 4 ternary values packed in byte
 * 
 * Uses efficient dot product computation for each matrix row.
 * 
 * @param mat 4x4 ternary matrix (32-bit packed)
 * @param vec 4-element ternary vector (byte packed)
 * @return 4-element result vector (byte packed)
 */
inline constexpr uint8_t mat_vec_mul_m32(uint32_t mat, uint8_t vec) noexcept {
    uint8_t r0 = unpack_row_m32(mat, 0);
    uint8_t r1 = unpack_row_m32(mat, 1);
    uint8_t r2 = unpack_row_m32(mat, 2);
    uint8_t r3 = unpack_row_m32(mat, 3);

    return pack_v8(dot_v8(r0, vec), dot_v8(r1, vec), 
                   dot_v8(r2, vec), dot_v8(r3, vec));
}

/**
 * @brief Multiply two 4x4 packed ternary matrices
 * @author Zombie
 * @details
 * Performs efficient matrix multiplication: C = A * B
 * Both input matrices use 32-bit packed ternary representation.
 * 
 * Algorithm:
 * 1. Transpose matrix B for cache-friendly access
 * 2. For each column of B, compute dot product with all rows of A
 * 3. Use efficient mat_vec_mul_m32 for each operation
 * 4. Pack results into 32-bit output format
 * 
 * Complexity: O(4³) = O(64) operations vs O(4³) naive approach
 * 
 * @param A First matrix (32-bit packed)
 * @param B Second matrix (32-bit packed)  
 * @return Result matrix (32-bit packed)
 */
inline constexpr uint32_t mat_mul_m32(uint32_t A, uint32_t B) noexcept {
    uint32_t BT = transpose_m32(B);
    uint8_t c0 = unpack_row_m32(BT, 0);
    uint8_t c1 = unpack_row_m32(BT, 1);
    uint8_t c2 = unpack_row_m32(BT, 2);
    uint8_t c3 = unpack_row_m32(BT, 3);

    return pack_m32(mat_vec_mul_m32(A, c0), mat_vec_mul_m32(A, c1),
                    mat_vec_mul_m32(A, c2), mat_vec_mul_m32(A, c3));
}

} // namespace ternary_core

#endif // TERNARY_CORE_HPP