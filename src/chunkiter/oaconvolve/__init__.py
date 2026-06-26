"""
Streaming overlap-add and sparse (multirate) convolution utilities.
"""

import itertools
import numpy as np
from scipy.signal import oaconvolve

__all__ = ['chunked_oaconvolve']

# ---------------------------------------------------------------------------
# 1. Standard streaming overlap-add convolution
# ---------------------------------------------------------------------------

def chunked_oaconvolve_singlerate(stream, kernel):
    """
    Convolve an infinite data stream with a finite kernel using overlap-add.

    Parameters
    ----------
    stream : generator of np.ndarray
        Yields 1-D input blocks of fixed length.
    kernel : np.ndarray
        1-D convolution kernel.

    Yields
    ------
    np.ndarray
        Output blocks. Most yielded blocks have the same length as the input
        blocks. After the input stream is exhausted, a final block of length
        ``len(kernel) - 1`` is yielded containing the remaining tail samples.
    """
    kernel = np.asarray(kernel)
    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional")

    M = kernel.shape[0]

    if M == 0:
        # Empty kernel: consume stream but produce no output.
        for _ in stream:
            pass
        return

    # Determine block size from the first yielded block.
    try:
        first_block = next(stream)
    except StopIteration:
        return

    first_block = np.asarray(first_block)
    if first_block.ndim != 1:
        raise ValueError("input blocks must be one-dimensional")

    L = first_block.shape[0]

    if L == 0:
        # Empty blocks: consume the rest of the stream and yield nothing.
        for _ in stream:
            pass
        return

    # Dtype that can hold both block and kernel values.
    dtype = np.result_type(first_block, kernel)
    overlap = np.zeros(M - 1, dtype=dtype)

    def _process(block):
        nonlocal overlap
        block = np.asarray(block)
        if block.shape[0] != L:
            raise ValueError(
                f"Block size changed: expected {L}, got {block.shape[0]}"
            )

        # Linear convolution of the current block with the kernel.
        y = oaconvolve(block, kernel, mode="full")
        # Ensure consistent dtype without unnecessary copy.
        y = y.astype(dtype, copy=False)

        if M > 1:
            # Add tail from previous block to the start of this block's output.
            y[: M - 1] += overlap
            # Save the new tail for the next block.
            overlap = y[L : L + M - 1].copy()
            # Return the fully-resolved samples.
            return y[:L]
        else:
            # M == 1: no overlap needed.
            return y

    yield _process(first_block)

    for block in stream:
        yield _process(block)

    # Flush the remaining overlap after the stream ends.
    if M > 1:
        yield overlap


# ---------------------------------------------------------------------------
# 2. Sparse (multirate) streaming convolution
# ---------------------------------------------------------------------------

def chunked_oaconvolve(stream, kernel, factor=1):
    """
    Convolve a high-rate stream with a low-rate kernel.

    The kernel is defined at a sample rate ``factor`` times lower than the
    stream.  The effective high-rate filter is the zero-inserted kernel.
    The implementation stores a polyphase state matrix of
    ``factor * (len(kernel) - 1)`` samples so that the upsampled kernel
    itself is never materialised.

    Parameters
    ----------
    stream : generator of np.ndarray
        Yields 1-D input blocks of fixed length at the high sample rate.
    kernel : np.ndarray
        1-D convolution kernel at the low sample rate.
    factor : int
        Ratio ``samplerate(stream) / samplerate(kernel)``.  Must be a
        positive integer.

    Yields
    ------
    np.ndarray
        Output blocks at the high sample rate.  Most blocks have the same
        length as the input blocks.  After the stream ends, a final tail
        block of length ``(len(kernel) - 1) * factor`` is yielded.
    """
    kernel = np.asarray(kernel)
    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional")

    M = kernel.shape[0]
    if M == 0:
        for _ in stream:
            pass
        return

    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")

    # Determine block size from the first yielded block.
    try:
        first_block = next(stream)
    except StopIteration:
        return

    first_block = np.asarray(first_block)
    if first_block.ndim != 1:
        raise ValueError("input blocks must be one-dimensional")

    L = first_block.shape[0]
    if L == 0:
        for _ in stream:
            pass
        return

    dtype = np.result_type(first_block, kernel)
    kernel = kernel.astype(dtype, copy=False)

    if factor == 1:
        # Delegate to the standard overlap-add implementation.
        yield from chunked_oaconvolve_singlerate(itertools.chain([first_block], stream), kernel)
        return

    I = factor

    if M > 1:
        states = np.zeros((I, M - 1), dtype=dtype)
    else:
        states = None

    offset = 0

    def _process_block(x):
        nonlocal offset
        x = np.asarray(x)
        if x.shape[0] != L:
            raise ValueError(
                f"Block size changed: expected {L}, got {x.shape[0]}"
            )
        x = x.astype(dtype, copy=False)
        y = np.zeros(L, dtype=dtype)

        pos = 0
        while pos < L:
            end = min(pos + I, L)
            chunk = x[pos:end]
            C = end - pos

            # Phases for this chunk: offset, offset+1, ..., offset+C-1 (mod I).
            # pos is always a multiple of I, so (offset + pos) % I == offset.
            perm = (offset + np.arange(C)) % I

            if M == 1:
                y[pos:end] = kernel[0] * chunk
            else:
                S = states[perm]  # C x (M-1)
                out_chunk = S[:, 0] + kernel[0] * chunk
                y[pos:end] = out_chunk

                new_S = np.empty_like(S)
                if M > 2:
                    new_S[:, : M - 2] = (
                        S[:, 1:] + chunk[:, np.newaxis] * kernel[1 : M - 1]
                    )
                new_S[:, M - 2] = kernel[M - 1] * chunk
                states[perm] = new_S

            pos = end

        offset = (offset + L) % I
        return y

    yield _process_block(first_block)

    for block in stream:
        yield _process_block(block)

    if M > 1:
        # Yield the remaining states, but start with the phase indicated by
        # ``offset`` so that the interleaved tail aligns with the output grid.
        yield np.roll(states, -offset, axis=0).T.ravel()


# ============================ Tests =======================================

def _make_stream(arr, chunk_size):
    """Helper to split an array into fixed-size chunks (drops any partial final block)."""
    n = (len(arr) // chunk_size) * chunk_size
    for i in range(0, n, chunk_size):
        yield arr[i : i + chunk_size]


# -- Tests for chunked_oaconvolve_singlerate -------------------------------------------


def test_standard_small_default():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    h = rng.standard_normal(50)
    chunk_size = 100

    out = np.concatenate(list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h)))
    expected = oaconvolve(x, h, mode="full")
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_standard_various_chunk_sizes():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(10000)
    h = rng.standard_normal(100)

    for chunk_size in [1, 10, 99, 100, 101, 500, 10000]:
        n = (len(x) // chunk_size) * chunk_size
        out = np.concatenate(
            list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h))
        )
        expected = oaconvolve(x[:n], h, mode="full")
        np.testing.assert_allclose(
            out,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed for chunk_size={chunk_size}",
        )


def test_standard_kernel_larger_than_block():
    rng = np.random.default_rng(456)
    x = rng.standard_normal(500)
    h = rng.standard_normal(200)

    for chunk_size in [1, 50, 100, 199, 200, 201, 500]:
        n = (len(x) // chunk_size) * chunk_size
        out = np.concatenate(
            list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h))
        )
        expected = oaconvolve(x[:n], h, mode="full")
        np.testing.assert_allclose(
            out,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed for chunk_size={chunk_size}",
        )


def test_standard_complex_dtype():
    rng = np.random.default_rng(789)
    x = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    h = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    chunk_size = 50

    out = np.concatenate(list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h)))
    expected = oaconvolve(x, h, mode="full")
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_standard_single_tap_kernel():
    rng = np.random.default_rng(111)
    x = rng.standard_normal(1000)
    h = np.array([2.5])
    chunk_size = 100

    out = np.concatenate(list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h)))
    expected = oaconvolve(x, h, mode="full")
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_standard_large_array():
    rng = np.random.default_rng(222)
    # ~40 MB of float64 data, comfortably below 16 GB limit.
    n = 5_000_000
    x = rng.standard_normal(n)
    h = rng.standard_normal(1000)
    chunk_size = 100_000

    out = np.concatenate(list(chunked_oaconvolve_singlerate(_make_stream(x, chunk_size), h)))
    expected = oaconvolve(x, h, mode="full")
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-10)


def test_standard_empty_kernel():
    rng = np.random.default_rng(333)
    x = rng.standard_normal(100)
    h = np.array([])

    out = list(chunked_oaconvolve_singlerate(_make_stream(x, 10), h))
    assert out == []


def test_standard_empty_stream():
    rng = np.random.default_rng(444)
    h = rng.standard_normal(10)

    def empty_stream():
        # Immediately exhausted generator.
        if False:
            yield np.array([])

    out = list(chunked_oaconvolve_singlerate(empty_stream(), h))
    assert out == []


def test_standard_single_block():
    rng = np.random.default_rng(555)
    x = rng.standard_normal(100)
    h = rng.standard_normal(20)

    def stream():
        yield x

    out = np.concatenate(list(chunked_oaconvolve_singlerate(stream(), h)))
    expected = oaconvolve(x, h, mode="full")
    np.testing.assert_allclose(out, expected, rtol=1e-10)


# -- Tests for chunked_oaconvolve ------------------------------------------

def _reference_sparse(x, kernel, factor):
    """Reference convolution using oaconvolve with upsampled kernel."""
    M = len(kernel)
    if M == 0:
        return np.array([], dtype=np.result_type(x, kernel))
    h_up = np.zeros((M - 1) * factor + 1, dtype=np.result_type(x, kernel))
    h_up[::factor] = kernel
    return oaconvolve(x, h_up, mode="full")


def test_sparse_small_default():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    h = rng.standard_normal(50)
    factor = 10
    chunk_size = 100

    out = np.concatenate(
        list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
    )
    expected = _reference_sparse(x, h, factor)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_sparse_various_chunk_sizes():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(10000)
    h = rng.standard_normal(100)
    factor = 10

    for chunk_size in [1, 10, 99, 100, 101, 500, 10000]:
        n = (len(x) // chunk_size) * chunk_size
        out = np.concatenate(
            list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
        )
        expected = _reference_sparse(x[:n], h, factor)
        np.testing.assert_allclose(
            out,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed for chunk_size={chunk_size}",
        )


def test_sparse_various_factors():
    rng = np.random.default_rng(456)
    x = rng.standard_normal(5000)
    h = rng.standard_normal(20)
    chunk_size = 100

    for factor in [1, 2, 3, 10, 99, 100, 101, 500]:
        n = (len(x) // chunk_size) * chunk_size
        out = np.concatenate(
            list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
        )
        expected = _reference_sparse(x[:n], h, factor)
        np.testing.assert_allclose(
            out,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed for factor={factor}",
        )


def test_sparse_factor_larger_than_block():
    rng = np.random.default_rng(789)
    x = rng.standard_normal(1000)
    h = rng.standard_normal(50)
    factor = 200
    chunk_size = 100

    n = (len(x) // chunk_size) * chunk_size
    out = np.concatenate(
        list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
    )
    expected = _reference_sparse(x[:n], h, factor)
    np.testing.assert_allclose(out, expected, rtol=2e-10)


def test_sparse_kernel_larger_than_block():
    rng = np.random.default_rng(321)
    x = rng.standard_normal(500)
    h = rng.standard_normal(200)
    factor = 5

    for chunk_size in [1, 50, 100, 199, 200, 201, 500]:
        n = (len(x) // chunk_size) * chunk_size
        out = np.concatenate(
            list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
        )
        expected = _reference_sparse(x[:n], h, factor)
        np.testing.assert_allclose(
            out,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed for chunk_size={chunk_size}",
        )


def test_sparse_complex_dtype():
    rng = np.random.default_rng(654)
    x = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    h = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    factor = 7
    chunk_size = 50

    out = np.concatenate(
        list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
    )
    expected = _reference_sparse(x, h, factor)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_sparse_single_tap_kernel():
    rng = np.random.default_rng(111)
    x = rng.standard_normal(1000)
    h = np.array([2.5])
    factor = 13
    chunk_size = 100

    out = np.concatenate(
        list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
    )
    expected = _reference_sparse(x, h, factor)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_sparse_large_array():
    rng = np.random.default_rng(222)
    # ~40 MB of float64 data, comfortably below 16 GB limit.
    n = 5_000_000
    x = rng.standard_normal(n)
    h = rng.standard_normal(100)
    factor = 10
    chunk_size = 100_000

    out = np.concatenate(
        list(chunked_oaconvolve(_make_stream(x, chunk_size), h, factor))
    )
    expected = _reference_sparse(x, h, factor)
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-10)


def test_sparse_empty_kernel():
    rng = np.random.default_rng(333)
    x = rng.standard_normal(100)
    h = np.array([])
    factor = 5

    out = list(chunked_oaconvolve(_make_stream(x, 10), h, factor))
    assert out == []


def test_sparse_empty_stream():
    rng = np.random.default_rng(444)
    h = rng.standard_normal(10)
    factor = 3

    def empty_stream():
        if False:
            yield np.array([])

    out = list(chunked_oaconvolve(empty_stream(), h, factor))
    assert out == []


def test_sparse_single_block():
    rng = np.random.default_rng(555)
    x = rng.standard_normal(100)
    h = rng.standard_normal(20)
    factor = 4

    def stream():
        yield x

    out = np.concatenate(list(chunked_oaconvolve(stream(), h, factor)))
    expected = _reference_sparse(x, h, factor)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


if __name__ == "__main__":
    # Standard tests
    test_standard_small_default()
    print("test_standard_small_default passed")

    test_standard_various_chunk_sizes()
    print("test_standard_various_chunk_sizes passed")

    test_standard_kernel_larger_than_block()
    print("test_standard_kernel_larger_than_block passed")

    test_standard_complex_dtype()
    print("test_standard_complex_dtype passed")

    test_standard_single_tap_kernel()
    print("test_standard_single_tap_kernel passed")

    test_standard_large_array()
    print("test_standard_large_array passed")

    test_standard_empty_kernel()
    print("test_standard_empty_kernel passed")

    test_standard_empty_stream()
    print("test_standard_empty_stream passed")

    test_standard_single_block()
    print("test_standard_single_block passed")

    # Sparse tests
    test_sparse_small_default()
    print("test_sparse_small_default passed")

    test_sparse_various_chunk_sizes()
    print("test_sparse_various_chunk_sizes passed")

    test_sparse_various_factors()
    print("test_sparse_various_factors passed")

    test_sparse_factor_larger_than_block()
    print("test_sparse_factor_larger_than_block passed")

    test_sparse_kernel_larger_than_block()
    print("test_sparse_kernel_larger_than_block passed")

    test_sparse_complex_dtype()
    print("test_sparse_complex_dtype passed")

    test_sparse_single_tap_kernel()
    print("test_sparse_single_tap_kernel passed")

    test_sparse_large_array()
    print("test_sparse_large_array passed")

    test_sparse_empty_kernel()
    print("test_sparse_empty_kernel passed")

    test_sparse_empty_stream()
    print("test_sparse_empty_stream passed")

    test_sparse_single_block()
    print("test_sparse_single_block passed")

    print("\nAll tests passed.")
