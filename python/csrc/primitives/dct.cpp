#include "primitives/dct.h"

#include <mlx/ops.h>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <tuple>
#include <vector>

namespace mlx_audio {

namespace {

// Cache key as tuple for simplicity (n_out, n_in, ortho)
using DCTCacheKey = std::tuple<int, int, bool>;

// Use map with emplace to work around array not having default ctor
std::map<DCTCacheKey, mlx::core::array> dct_cache;
std::mutex dct_cache_mutex;

}  // namespace

mlx::core::array get_dct_matrix(
    int n_out,
    int n_in,
    const std::string& norm,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    bool ortho = (norm == "ortho");

    // Check cache
    DCTCacheKey key = std::make_tuple(n_out, n_in, ortho);
    {
        std::lock_guard<std::mutex> lock(dct_cache_mutex);
        auto it = dct_cache.find(key);
        if (it != dct_cache.end()) {
            return it->second;
        }
    }

    // Compute DCT-II basis matrix
    // C[k, n] = cos(pi * k * (2n + 1) / (2N))
    // for k = 0, ..., n_out-1 and n = 0, ..., n_in-1

    // Create index arrays
    auto k = arange(n_out, s);  // (n_out,)
    auto n = arange(n_in, s);   // (n_in,)

    // Reshape for broadcasting: k -> (n_out, 1), n -> (1, n_in)
    k = reshape(k, {n_out, 1}, s);
    n = reshape(n, {1, n_in}, s);

    // Compute angles: pi * k * (2n + 1) / (2 * n_in)
    auto two_n_plus_one = add(multiply(n, array(2.0f, float32), s), array(1.0f, float32), s);
    auto angles = multiply(
        multiply(array(static_cast<float>(M_PI), float32), astype(k, float32, s), s),
        divide(astype(two_n_plus_one, float32, s), array(2.0f * n_in, float32), s),
        s
    );

    // Compute cosine
    auto dct_basis = cos(angles, s);

    // Apply orthonormal scaling if requested
    if (ortho) {
        // Scale first row by 1/sqrt(n_in), rest by sqrt(2/n_in)
        float scale_0 = 1.0f / std::sqrt(static_cast<float>(n_in));
        float scale_rest = std::sqrt(2.0f / static_cast<float>(n_in));

        if (n_out > 0) {
            // Get first row and scale
            auto row_0 = slice(dct_basis, {0, 0}, {1, n_in}, s);
            row_0 = multiply(row_0, array(scale_0, float32), s);

            if (n_out > 1) {
                // Get rest of rows and scale
                auto rows_rest = slice(dct_basis, {1, 0}, {n_out, n_in}, s);
                rows_rest = multiply(rows_rest, array(scale_rest, float32), s);

                // Concatenate
                dct_basis = concatenate({row_0, rows_rest}, 0, s);
            } else {
                dct_basis = row_0;
            }
        }
    }

    // Ensure float32
    dct_basis = astype(dct_basis, float32, s);

    // Cache the result using emplace to avoid default constructor requirement
    {
        std::lock_guard<std::mutex> lock(dct_cache_mutex);
        dct_cache.emplace(key, dct_basis);
    }

    return dct_basis;
}

mlx::core::array dct(
    const mlx::core::array& x,
    int n,
    int axis,
    const std::string& norm,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate norm
    if (norm != "ortho" && norm != "") {
        throw std::invalid_argument("norm must be 'ortho' or empty string");
    }

    // Normalize axis
    int ndim = x.ndim();
    if (axis < 0) {
        axis = ndim + axis;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument("axis out of bounds");
    }

    // Get input size along axis
    int input_size = x.shape(axis);
    if (n <= 0) {
        n = input_size;
    }

    // Get DCT matrix: (n, input_size)
    auto dct_matrix = get_dct_matrix(n, input_size, norm, s);

    // Move axis to last position for matmul
    array y = x;
    if (axis != ndim - 1) {
        std::vector<int> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[axis], perm[ndim - 1]);
        y = transpose(y, perm, s);
    }

    // Apply DCT: output = x @ dct_matrix.T
    // dct_matrix shape: (n, input_size)
    // x shape: (..., input_size)
    // output shape: (..., n)
    auto result = matmul(y, transpose(dct_matrix, s), s);

    // Move axis back if needed
    if (axis != ndim - 1) {
        std::vector<int> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[axis], perm[ndim - 1]);
        result = transpose(result, perm, s);
    }

    return result;
}

}  // namespace mlx_audio
