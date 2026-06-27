"""
streaming sliding window extraction
"""

import itertools
import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = ['sliding_window']

def sliding_window(data, window, step, output_chunksize=None, padding=False, yield_remainder=True):
  """Extract sliding windows from a stream of 1D ndarray chunks.

  Takes an iterator of 1D ndarrays (a data stream) and yields 2D ndarrays
  where each row is a sliding window over the concatenated input data,
  advancing by *step* samples between consecutive windows.

  Args:
      data: Iterator yielding 1-D ``np.ndarray`` chunks.
      window (int or np.ndarray): Window size (int) or coefficient array
          applied element-wise to each window.
      step (int): Step size between consecutive windows.
      output_chunksize (int, optional): Number of windows per output chunk.
          If ``None``, chosen to approximately match input chunk memory
          (~1–4 MB).
      padding (bool): If ``True``, the last chunk is zero-padded to
          *output_chunksize* rows.  All chunks are yielded as
          ``(actual_size, windows)`` tuples.
      yield_remainder (bool): If ``True`` (default), the last partial chunk
          is yielded even if it has fewer than *output_chunksize* windows.

  Yields:
      np.ndarray or (int, np.ndarray): 2-D array of shape
      ``(n_windows, window_size)``.  With ``padding=True``, yields
      ``(actual_n_windows, windows)`` for every chunk.

  Raises:
      ValueError: If *window* size or *step* is less than 1, or if
          *output_chunksize* is 0.

  Example:
      >>> import numpy as np
      >>> from chunkiter.sliding_window import sliding_window
      >>> source = [np.array([1., 2., 3., 4., 5., 6., 7.])]
      >>> for chunk in sliding_window(source, 3, 2, output_chunksize=2):
      ...     print(chunk)
      [[1. 2. 3.]
       [3. 4. 5.]]
      [[5. 6. 7.]]
  """

  # --- normalize window ---
  if isinstance(window, (int, np.integer)):
    window_size = int(window)
    window_coeffs = None
  else:
    window_coeffs = np.asarray(window)
    window_size = window_coeffs.size

  if window_size < 1:
    raise ValueError("window size must be positive")
  if step < 1:
    raise ValueError("step must be positive")

  # --- determine output_chunksize if not given ---
  if output_chunksize is None:
    first_chunk = next(data)
    data = itertools.chain([first_chunk], data)
    input_chunk_bytes = first_chunk.nbytes
    itemsize = first_chunk.dtype.itemsize

    candidate = max(1, int(input_chunk_bytes / (window_size * itemsize)))
    max_cs = max(1, (4 * 1024**2) // (window_size * itemsize))
    min_cs = max(1, (1 * 1024**2) // (window_size * itemsize))
    output_chunksize = min(max(candidate, min_cs), max_cs)

  if output_chunksize < 1:
    raise ValueError("output_chunksize must be at least 1")

  # --- buffer state ---
  try:
    first_for_buffer = next(data)
  except StopIteration:
    return
  input_chunks = [first_for_buffer]
  input_start_i = 0
  input_stop_i = first_for_buffer.shape[0]
  output_pos = 0

  while True:
    # 1. discard input chunks fully behind output_pos
    while len(input_chunks) and output_pos > input_start_i + input_chunks[0].shape[0]:
      input_start_i += input_chunks[0].shape[0]
      input_chunks.pop(0)

    # 2. fetch more input chunks as needed
    needed_end = output_pos + window_size + (output_chunksize - 1) * step
    while len(input_chunks) == 0 or needed_end > input_stop_i:
      try:
        next_chunk = next(data)
        input_chunks.append(next_chunk)
        input_stop_i += next_chunk.shape[0]
      except StopIteration:
        break

    if len(input_chunks) == 0:
      break

    # 3. how many windows can we extract?
    available_end = min(input_stop_i, needed_end)
    available_len = available_end - output_pos

    if available_len < window_size:
      break

    n_windows = (available_len - window_size) // step + 1
    n_windows = min(n_windows, output_chunksize)

    if n_windows == 0:
      break

    # 4. extract views from the input buffer
    raw_needed = window_size + (n_windows - 1) * step
    views = []
    skip = output_pos - input_start_i
    collected = 0

    for chunk in input_chunks:
      rel = chunk[skip:, ...]
      rel = rel[:raw_needed - collected, ...]
      collected += rel.shape[0]
      skip = 0
      if rel.shape[0] > 0:
        views.append(rel)
      if collected >= raw_needed:
        break

    arr = views[0] if len(views) == 1 else np.concatenate(views)

    # 5. extract windows via as_strided (step-skipping along axis 0)
    strides = (arr.strides[0] * step, arr.strides[0])
    windows = as_strided(arr, shape=(n_windows, window_size), strides=strides)

    if window_coeffs is not None:
      windows = windows * window_coeffs
    else:
      windows = windows.copy()

    # 6. yield (with optional padding / remainder handling)
    if padding:
      actual_size = n_windows
      if n_windows < output_chunksize:
        pad_shape = (output_chunksize - n_windows, window_size)
        windows = np.concatenate([windows, np.zeros(pad_shape, dtype=windows.dtype)], axis=0)
      if actual_size == output_chunksize or yield_remainder:
        yield actual_size, windows
    else:
      if n_windows == output_chunksize or yield_remainder:
        yield windows
      else:
        break

    # 7. advance
    output_pos += n_windows * step


# ============================================================
# --- tests ---
# ============================================================

# --- helper --------------------------------------------------
def _ref_windows(x, window_size, step, coeffs=None):
  """Build reference sliding windows from a full 1D array *x*."""
  ws = []
  for i in range(0, len(x) - window_size + 1, step):
    w = x[i:i + window_size]
    if coeffs is not None:
      w = w * coeffs
    ws.append(w)
  if not ws:
    return np.empty((0, window_size), dtype=x.dtype)
  return np.stack(ws)

def _split_random(rng, x, n_splits):
  """Split *x* into *n_splits*+1 chunks at random boundaries."""
  boundaries = sorted(rng.choice(range(1, len(x)), size=n_splits, replace=False))
  return np.split(x, boundaries) if boundaries else [x]

# --- basic correctness ---------------------------------------

def test_basic():
  """Varying chunks, int window, normal step, no padding."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=5)), axis=0)
  assert np.allclose(result, ref)

def test_with_coefficients():
  """ndarray window, normal step."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  coeffs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step, coeffs)
  result = np.concatenate(list(sliding_window(iter(chunks), coeffs, step, output_chunksize=5)), axis=0)
  assert np.allclose(result, ref)

def test_step_larger_than_window():
  """step > window_size places non-overlapping contiguous regions."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 3, 7
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=3)), axis=0)
  assert np.allclose(result, ref)

def test_step_equals_window():
  """step == window_size: non-overlapping, abutting windows."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 10, 10
  chunks = _split_random(rng, x, 12)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=4)), axis=0)
  assert np.allclose(result, ref)

def test_step_one():
  """step == 1: maximum overlap, every possible window."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(200)
  window, step = 10, 1
  chunks = _split_random(rng, x, 15)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=23)), axis=0)
  assert np.allclose(result, ref)

# --- single / tiny inputs ------------------------------------

def test_empty_input():
  """Empty iterator yields nothing."""
  import numpy as np
  result = list(sliding_window(iter([]), 3, 2, output_chunksize=4))
  assert result == []

def test_input_shorter_than_window():
  """Total data shorter than window_size yields nothing."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(4)
  chunks = [x[:2], x[2:]]
  result = list(sliding_window(iter(chunks), 5, 2, output_chunksize=3))
  assert len(result) == 0

def test_exactly_one_window():
  """Exactly window_size samples produce a single window."""
  import numpy as np
  rng = np.random.default_rng(42)
  window, step = 10, 5
  x = rng.standard_normal(window)
  chunks = [x[:3], x[3:7], x[7:]]
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=4)), axis=0)
  # 1 window expected
  assert result.shape == (1, window)
  assert np.allclose(result, ref)

def test_single_chunk_input():
  """Single input chunk (no splits at all)."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(200)
  window, step = 11, 7
  chunks = [x]
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=5)), axis=0)
  assert np.allclose(result, ref)

def test_single_element_chunks():
  """Every input chunk is size 1 — stress test for buffer management."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(500)
  window, step = 7, 3
  chunks = [np.array([v]) for v in x]
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=11)), axis=0)
  assert np.allclose(result, ref)

# --- varying chunk sizes (deterministic) ---------------------

def test_varying_chunk_sizes():
  """Irregular but deterministic chunk sizes."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(1000)
  window, step = 17, 11
  sizes = [3, 47, 8, 91, 1, 200, 5, 33, 100, 512]
  chunks = [x[sum(sizes[:i]):sum(sizes[:i]) + s] for i, s in enumerate(sizes)]
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=7)), axis=0)
  assert np.allclose(result, ref)

# --- auto output_chunksize -----------------------------------

def test_output_chunksize_auto():
  """output_chunksize=None picks a sensible value ~1-4 MB."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(10000)
  window, step = 5, 3
  chunks = np.split(x, 10)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step)), axis=0)
  assert np.allclose(result, ref)

# --- exact multiple (no remainder) ---------------------------

def test_exact_multiple():
  """output_chunksize divides total windows exactly — no partial last chunk."""
  import numpy as np
  rng = np.random.default_rng(42)
  window, step = 5, 2
  output_chunksize = 10
  # total length so that (n-w)//s + 1 is a multiple of output_chunksize
  # 100 → (100-5)//2+1 = 48, not a multiple of 10
  # 105 → (105-5)//2+1 = 51, not a multiple of 10
  # 104 → (104-5)//2+1 = 50 ✓
  total_len = 104
  x = rng.standard_normal(total_len)
  chunks = _split_random(rng, x, 8)
  ref = _ref_windows(x, window, step)
  result = np.concatenate(list(sliding_window(iter(chunks), window, step, output_chunksize=output_chunksize)), axis=0)
  assert np.allclose(result, ref)
  # every chunk should be full size
  for c in sliding_window(iter(chunks), window, step, output_chunksize=output_chunksize):
    assert c.shape[0] == output_chunksize

# --- padding -------------------------------------------------

def test_padding():
  """padding=True: all chunks become (actual_size, padded_array)."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step)
  csize = 13

  result = list(sliding_window(iter(chunks), window, step, output_chunksize=csize, padding=True))
  assert len(result) > 0
  assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

  actual_sizes = [c[0] for c in result]
  padded_arrays = [c[1] for c in result]
  for a in padded_arrays:
    assert a.shape == (csize, window)

  # reconstruct
  unpadded = [p[:a] for (a, p) in result]
  reconstructed = np.concatenate(unpadded, axis=0)
  assert np.allclose(reconstructed, ref)

  for s in actual_sizes[:-1]:
    assert s == csize

def test_padding_first_is_full():
  """When the first (and only) chunk is already full, padding still
     yields a (full_size, array) tuple — no spurious padding."""
  import numpy as np
  rng = np.random.default_rng(42)
  window, step = 4, 2
  csize = 3
  # need enough data for exactly csize windows: (csize-1)*step + window = 8
  x = rng.standard_normal(8)
  chunks = [x[:3], x[3:6], x[6:]]
  ref = _ref_windows(x, window, step)
  # only one output chunk, should be full
  result = list(sliding_window(iter(chunks), window, step, output_chunksize=csize, padding=True))
  assert len(result) == 1
  actual, arr = result[0]
  assert actual == csize
  assert arr.shape == (csize, window)
  assert np.allclose(arr, ref)

def test_coefficients_with_padding():
  """ndarray coefficients combined with padding=True."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  coeffs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step, coeffs)
  csize = 13

  result = list(sliding_window(iter(chunks), coeffs, step, output_chunksize=csize, padding=True))
  assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

  unpadded = [p[:a] for (a, p) in result]
  reconstructed = np.concatenate(unpadded, axis=0)
  assert np.allclose(reconstructed, ref)

  # padded rows should be all zeros
  _, last = result[-1]
  if result[-1][0] < csize:
    assert np.all(last[result[-1][0]:] == 0)

def test_padding_yield_remainder_false():
  """padding=True + yield_remainder=False: drop partial last chunk."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step)
  csize = 9

  result = list(sliding_window(iter(chunks), window, step, output_chunksize=csize,
                               padding=True, yield_remainder=False))
  assert all(isinstance(c, tuple) and len(c) == 2 for c in result)
  # every yielded chunk must be full
  for actual_size, arr in result:
    assert actual_size == csize

  unpadded = [p[:a] for (a, p) in result]
  reconstructed = np.concatenate(unpadded, axis=0)
  assert np.allclose(reconstructed, ref[:reconstructed.shape[0]])

# --- yield_remainder=False -----------------------------------

def test_yield_remainder_false():
  """yield_remainder=False: last partial chunk is suppressed."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step)

  result = np.concatenate(
    list(sliding_window(iter(chunks), window, step, output_chunksize=13, yield_remainder=False)),
    axis=0,
  )
  assert result.shape[0] % 13 == 0
  assert np.allclose(result, ref[:result.shape[0]])

def test_coefficients_yield_remainder_false():
  """ndarray coefficients + yield_remainder=False."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(100)
  window, step = 5, 3
  coeffs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
  chunks = _split_random(rng, x, 9)
  ref = _ref_windows(x, window, step, coeffs)

  result = np.concatenate(
    list(sliding_window(iter(chunks), coeffs, step, output_chunksize=15, yield_remainder=False)),
    axis=0,
  )
  assert result.shape[0] % 15 == 0
  assert np.allclose(result, ref[:result.shape[0]])

# --- dtype handling ------------------------------------------

def test_dtype_preservation():
  """Input dtype is preserved in output when no coefficients are used."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(200).astype(np.float32)
  window, step = 7, 3
  chunks = _split_random(rng, x, 10)
  for c in sliding_window(iter(chunks), window, step, output_chunksize=5):
    assert c.dtype == np.float32

def test_coefficient_dtype_float32_in_float64_window():
  """float32 input with float64 coefficients → float64 output (numpy upcast)."""
  import numpy as np
  rng = np.random.default_rng(42)
  x = rng.standard_normal(200).astype(np.float32)
  window, step = 7, 3
  coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
  chunks = _split_random(rng, x, 15)
  ref = _ref_windows(x.astype(np.float64), window, step, coeffs)
  result = np.concatenate(list(sliding_window(iter(chunks), coeffs, step, output_chunksize=7)), axis=0)
  assert result.dtype == np.float64
  assert np.allclose(result, ref)

# --- validation error paths ----------------------------------

def test_bad_window_size():
  """window size < 1 raises ValueError."""
  import numpy as np
  try:
    next(sliding_window(iter([np.ones(10)]), 0, 2, output_chunksize=3))
    assert False, "expected ValueError"
  except ValueError:
    pass

def test_bad_step():
  """step < 1 raises ValueError."""
  import numpy as np
  try:
    next(sliding_window(iter([np.ones(10)]), 3, 0, output_chunksize=3))
    assert False, "expected ValueError"
  except ValueError:
    pass

# --- timing comparison ---------------------------------------

def test_timing():
  """Compare wall-clock throughput of sliding_window vs rechunk.

  Both process ~400 MB of float64 data split into 100 chunks of ~4 MB each,
  using 50 % overlap (chunk_size=1024, overlap_size=512 for rechunk;
  window=1024, step=512 for sliding_window).
  """
  import numpy as np
  from chunkiter.functions import rechunk
  import time

  rng = np.random.default_rng(42)
  chunk_size = 524288          # 4 MB for float64
  n_chunks = 100
  total = chunk_size * n_chunks      # 52 428 800 ≈ 419 MB
  x = rng.standard_normal(total)
  chunks = np.split(x, n_chunks)

  window, step = 1024, 512
  overlap = step                 # 50 % overlap for rechunk

  chunk_mb = chunks[0].nbytes / 1024**2
  total_mb = x.nbytes / 1024**2
  n_windows = (total - window) // step + 1
  print()
  print(f"--- Timing comparison (50 % overlap) ---")
  print(f"  data:    {total} float64 elements ({total_mb:.0f} MB total)")
  print(f"  input:   {n_chunks} chunks of ~{chunk_mb:.1f} MB each")
  print(f"  windows: {n_windows} of size {window}, step {step}")

  # --- rechunk: chunk_size=1024, overlap=512 (50% overlap) ---
  t0 = time.perf_counter()
  rechunked = list(rechunk(iter(chunks), window, overlap_size=overlap))
  t1 = time.perf_counter()
  rt = t1 - t0
  r_out_mb = sum(c.nbytes for c in rechunked) / 1024**2
  print(f"  rechunk:         {rt:.4f}s  ({total_mb:.0f} MB → {r_out_mb:.0f} MB,  {total_mb/rt:.0f} MB/s in)")

  # --- sliding_window: window=1024, step=512 (50% overlap) ---
  t0 = time.perf_counter()
  sw_result = np.concatenate(list(sliding_window(iter(chunks), window, step)), axis=0)
  t1 = time.perf_counter()
  st = t1 - t0
  s_out_mb = sw_result.nbytes / 1024**2
  print(f"  sliding_window:  {st:.4f}s  ({total_mb:.0f} MB → {s_out_mb:.0f} MB,  {total_mb/st:.0f} MB/s in)")

  # verify correctness
  ref = _ref_windows(x, window, step)
  assert np.allclose(sw_result, ref)

  # rechunk output concatenated should match flattened windows
  ref_flat = ref.ravel()
  rechunk_flat = np.concatenate([np.asarray(c) for c in rechunked])
  # rechunk has chunk_size non-overlapping + overlap parts — not trivially
  # the same as sliding_window output, so we just check sizes are plausible
  assert rechunk_flat.size > 0
  assert st > 0 and rt > 0
  print("  ✓ timing comparison ok")


if __name__ == "__main__":
  tests = [test_basic, test_with_coefficients, test_step_larger_than_window,
           test_step_equals_window, test_step_one,
           test_empty_input, test_input_shorter_than_window,
           test_exactly_one_window, test_single_chunk_input,
           test_single_element_chunks, test_varying_chunk_sizes,
           test_output_chunksize_auto, test_exact_multiple,
           test_padding, test_padding_first_is_full,
           test_coefficients_with_padding, test_padding_yield_remainder_false,
           test_yield_remainder_false, test_coefficients_yield_remainder_false,
           test_dtype_preservation, test_coefficient_dtype_float32_in_float64_window,
           test_bad_window_size, test_bad_step,
           test_timing]

  for t in tests:
    t()
    print(f"{t.__name__} passed")

  print(f"\nAll {len(tests)} tests passed.")
