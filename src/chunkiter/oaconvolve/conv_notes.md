# Notes on `conv.py` and `conv.tex`

## Goal

Create a **streaming, memory-efficient convolution module** for an infinite input stream with limited RAM. The module provides two variants:

1. Standard overlap-add convolution for a kernel at the same sample rate.
2. Sparse (multirate) convolution for a kernel defined at a lower sample rate (`factor` times lower than the signal).

## Steps

### 1. Initial implementation of standard overlap-add

**File:** `testconv.py`

Created `chunked_oaconvolve(stream, kernel)`.

**Key logic:**
- Pulls the first block from the generator to auto-detect the fixed block size `L`.
- Uses `scipy.signal.oaconvolve(..., mode="full")` on each block to compute linear convolution.
- Maintains an overlap buffer of length `M - 1` (`M` = kernel length) between blocks.
- Adds the previous overlap to the beginning of the current block output, then emits the fully-resolved first `L` samples. Saves the new tail for the next block.
- After the stream ends, flushes the remaining overlap buffer.

**Test fixes:**
- `_make_stream` helper had to drop the final partial block to guarantee fixed-size chunks, otherwise the block-size assertion triggered.

### 2. Extension to sparse / multirate convolution

**File:** `testconv2.py`

Created `chunked_oaconvolve2(stream, kernel, factor)`.

**Motivation:** For a low-frequency kernel at a high sample rate, naively zero-inserting the kernel would produce an enormous FIR (length `(M-1)*factor + 1`) and an equally huge overlap buffer. This is infeasible for large `factor` (e.g. `factor=200`).

**Key logic (polyphase approach):**
- Instead of materialising the upsampled kernel, store a polyphase state matrix of shape `(factor, M - 1)`.
- Each sample `x[n]` belongs to phase `p = n % factor`. Only that phase’s state row is updated.
- Output for a sample: `y[n] = state[p, 0] + kernel[0] * x[n]`.
- State update (shift-and-add): `state[p, j] = state[p, j+1] + kernel[j+1] * x[n]` for `j = 0..M-3`, and `state[p, M-2] = kernel[M-1] * x[n]`.
- `offset` tracks the phase of the next expected sample across blocks.

**Critical bug fixes:**
- **Tail flush ordering:** After the stream ends, the remaining state matrix must be emitted as a tail of length `(M-1)*factor`. The rows must be cyclically shifted by `-offset` so the phase ordering aligns with the high-rate output grid: `np.roll(states, -offset, axis=0).T.ravel()`. Without this, the tail samples are interleaved in the wrong order, causing large mismatches at block boundaries.
- **Tolerance:** One test (`factor_larger_than_block`, `factor=200`) had a single-element mismatch at machine epsilon level (`~1.2e-15` absolute, `~1.2e-10` relative). Relaxed `rtol` from `1e-10` to `2e-10` for that case.

### 3. Merge into single module

**Files:** Merged `testconv.py` and `testconv2.py` into `conv.py`. Removed the old files.

**Renamed functions:**
- `chunked_oaconvolve` → `chunked_oaconvolve_singlerate`
- `chunked_oaconvolve2` → `chunked_oaconvolve(stream, kernel, factor=1)`

When `factor == 1`, the sparse variant delegates to the standard one.

All tests renamed with `test_standard_*` and `test_sparse_*` prefixes and verified passing.

### 4. Mathematical write-up

**File:** `conv.tex`

Wrote a LaTeX document explaining both algorithms:
- Standard overlap-add with block-wise partition and overlap buffer.
- Polyphase decomposition for the sparse case.
- Derivation of the state matrix update equations.
- Memory-footprint comparison table.

### 5. Generalisation cleanup

- Removed application-specific language (laser, RIN, noise, bandpass examples) from the `.tex` file and docstrings to make the module purely generic.
- Fixed a nonsensical docstring in `chunked_oaconvolve` that compared `factor * (M-1)` and `(M-1) * factor` as if they were different. Rewrote to state that the polyphase state matrix stores `factor * (M-1)` samples and the upsampled kernel is never materialised.

## Final file structure

```
conv.py    # Streaming overlap-add and sparse convolution with tests
conv.tex   # LaTeX documentation of the algorithms
```

## Key invariants to preserve

- The stream generator **must** yield fixed-size blocks. The block size is determined from the first block and enforced for all subsequent blocks.
- Empty kernel → empty output (consume stream, yield nothing).
- Empty stream → empty output.
- The sparse tail flush **must** roll the state matrix by `-offset` before flattening.
- Test helpers (`_make_stream`) must drop partial final blocks to keep block size constant.
