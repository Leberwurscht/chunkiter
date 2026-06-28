import itertools, uuid

import numpy as np
import scipy.signal

from .functions import rechunk, cache

def mean(iterator):
  """Compute the mean along the first dimension of a chunk iterator.

  Accumulates sum and count across chunks without ever holding all data
  in memory.

  Args:
      iterator: Iterator yielding np.ndarray chunks.

  Returns:
      np.ndarray: Mean along axis 0.

  Example:
      >>> import numpy as np
      >>> chunks = [np.array([1., 2., 3.]), np.array([4., 5., 6.])]
      >>> chunkiter.mean(chunks)
      3.5
  """
  s = 0
  n = 0
  for d in iterator:
    s += np.sum(d,axis=0)
    n += d.shape[0]
  return s/n

def mix(data, frq, samplerate=1):
  frequency = (frq*np.ones(chunksize, dtype=float) for i in itertools.count())

  deltat = samplerate**-1
  last_phase = 0

  for data_chunk in data:
    relative_phase = np.cumsum(2*np.pi*frq*deltat*np.ones(data_chunk.size))
    relative_phase -= relative_phase[0]
    phase = relative_phase + last_phase
    yield np.angle(np.exp(1j*phase))*data_chunk
    last_phase = phase[-1]

def sum(iterator):
  """Compute the sum along the first dimension of a chunk iterator.

  Args:
      iterator: Iterator yielding np.ndarray chunks.

  Returns:
      np.ndarray: Element-wise sum along axis 0.

  Example:
      >>> chunks = [np.array([1., 2.]), np.array([3., 4.])]
      >>> chunkiter.sum(chunks)
      10.0
  """
  s = 0
  for d in iterator:
    s += np.sum(daxis=0)
  return s

def unwrap(iterator):
  """Apply ``np.unwrap`` along the first dimension of a chunk iterator.

  Correctly handles phase continuity across chunk boundaries.

  Args:
      iterator: Iterator yielding np.ndarray chunks.

  Yields:
      np.ndarray: Unwrapped chunks.

  Example:
      >>> import numpy as np
      >>> # Phase jumps between chunks handled correctly:
      >>> chunks = [np.array([3.1, 3.2, -3.0, -2.9])]
      >>> list(chunkiter.unwrap(chunks))
      [array([3.1, 3.2, 3.283..., 3.383...])]
  """
  switchover = None

  for d in iterator:
    if switchover is None:
      switchover = np.zeros((2,)+d.shape[1:], dtype=d.dtype)
      switchover[0,...] = d[0,...]

    switchover[1,...] = d[0,...]
    switchover = np.unwrap(switchover, axis=0)

    d[0,...] = switchover[1,...]

    d = np.unwrap(d, axis=0)
    yield d

    switchover[0,...] = d[-1,...]

def concatenate(iterator):
  """Concatenate all chunks of an iterator into a single numpy array.

  Loads the entire iterator into memory — only suitable when the total
  data fits in RAM.  For tuple chunks, concatenates each entry separately.

  Args:
      iterator: Iterator yielding np.ndarray chunks or tuples thereof.

  Returns:
      np.ndarray or tuple of np.ndarray: Fully concatenated data.

  Example:
      >>> chunks = [np.ones((3, 2)), np.ones((4, 2)) * 2]
      >>> chunkiter.concatenate(chunks)
      array([[1., 1.],
             [1., 1.],
             [1., 1.],
             [2., 2.],
             [2., 2.],
             [2., 2.],
             [2., 2.]])
  """

  first, iterator = peek(iterator)
  if type(first)==tuple:
    return tuple(np.concatenate(arrays,axis=0) for arrays in zip(*iterator))
  else:
    return np.concatenate(tuple(iterator),axis=0)

def _batchavg_chunk(data, batchsize):
  # batchavg for a single chunk
  batches = data.shape[0]//batchsize
  rest = data[batches*batchsize:,...]
  data = data[:batches*batchsize,...]
  batched_avgs = np.nanmean(data.reshape((batches,batchsize)+data.shape[1:]),axis=1)
  return batched_avgs, rest

def _batchavg(iterator, batchsize, allow_remainder=False):
  # batchavg of chunk iterator, without rechunking
  rest = None
  for d in iterator:
    if rest is not None: d = np.concatenate((rest, d), axis=0) # TODO: inefficient if batchsize>>input_chunksize
    batched_avgs, rest = _batchavg_chunk(d, batchsize)
    yield batched_avgs
  if rest.size and not allow_remainder: raise ValueError("batchavg: data left, if ok set allow_remainder=True")

def batchavg(iterator, batchsize, chunksize=None, allow_remainder=False):
  """Downsample a chunk iterator by computing batch averages.

  Replaces each consecutive group of *batchsize* samples along axis 0 with
  their mean, then rechunks back to the original chunk size.

  Args:
      iterator: Iterator yielding np.ndarray chunks.
      batchsize (int): Number of samples to average together.
      chunksize (int, optional): Output chunk size.  Defaults to the chunk
          size of the first input chunk.
      allow_remainder (bool): If ``False``, raises ``ValueError`` when the
          total sample count isn't divisible by *batchsize*.

  Yields:
      np.ndarray: Downsampled chunks.

  Raises:
      ValueError: If remainder exists and *allow_remainder* is ``False``.

  Example:
      >>> import numpy as np
      >>> chunks = [np.array([1., 2., 3., 4., 5., 6.])]
      >>> list(chunkiter.batchavg(chunks, 2))
      [array([1.5, 3.5, 5.5])]
  """
  iterator = iter(iterator)

  if chunksize is None:
    first = next(iterator)
    chunksize = first.shape[0]
    iterator = itertools.chain([first], iterator)

  yield from rechunk(_batchavg(iterator, batchsize, allow_remainder), chunksize)

def enumerate(iterator):
  """Enumerate chunks, yielding ``(counter_array, chunk)`` pairs.

  Unlike Python's ``enumerate`` which yields ``(int, chunk)``, this yields an
  integer array of sample indices matching the chunk's length, making it
  convenient for per-sample indexing in streaming pipelines.

  Args:
      iterator: Iterator yielding np.ndarray chunks.

  Yields:
      (np.ndarray, np.ndarray or tuple): Counter array and the corresponding chunk.

  Example:
      >>> import numpy as np
      >>> chunks = [np.ones(3), np.ones(2)]
      >>> for idx, chunk in chunkiter.enumerate(chunks):
      ...     print(idx, chunk)
      [0 1 2] [1. 1. 1.]
      [3 4] [1. 1.]
  """
  start = 0
  for chunk in iterator:
    n = chunk[0].shape[0] if type(chunk)==tuple else chunk.shape[0]
    counter = np.arange(n) + start
    yield counter, chunk
    start += n

def tee(iterator, n=2, max_buffer=1):
  """Memory-bounded ``itertools.tee`` for chunk iterators.

  Standard ``itertools.tee`` buffers ALL data, potentially filling RAM.
  This variant keeps at most *max_buffer* items and raises ``Exception``
  if consumers diverge too far.

  Args:
      iterator: Source iterator.
      n (int, optional): Number of output iterators.  If ``None``, returns a
          generator that yields an unlimited number of tee'd iterators.
      max_buffer (int): Maximum buffered items before raising ``Exception``.

  Returns:
      tuple or generator: *n* independent iterators (or an infinite generator
      if *n* is ``None``).

  Raises:
      Exception: If any consumer falls behind the buffer window.

  Example:
      >>> chunks = [np.array([1., 2.]), np.array([3., 4.])]
      >>> a, b = chunkiter.tee(chunks)
      >>> list(a)
      [array([1., 2.]), array([3., 4.])]
      >>> list(b)
      [array([1., 2.]), array([3., 4.])]
  """

  buffer = []
  buffer_start = 0
  next_offsets = {} # per-consumer next offsets to yield
  done = False

  def gen(i):
    nonlocal buffer, buffer_start, next_offsets, done

    next_offsets[i] = 0

    while True:
      read_offset = next_offsets[i]
      buffer_index = read_offset-buffer_start
#      print("======{}=====".format(i))
#      print("buffer {}".format(buffer))
#      print("buffer_start {}".format(buffer_start))
#      print("next_offsets {}".format(next_offsets))
#      print("done {}".format(done))
#      print()
#      print("read_offset {}".format(read_offset))
#      print("buffer_index {}".format(buffer_index))
#      print("buffer length {}".format(len(buffer)))

      # check if buffer needs to be advanced
      if buffer_index>=len(buffer):
        try:
          item = next(iterator)
        except StopIteration:
          done = True
#          print("DONE")
        else:
          buffer_length_before = len(buffer)
          buffer.append(item)
          buffer = buffer[-max_buffer:]
          buffer_length_after = len(buffer)
          buffer_start += 1 - (buffer_length_after-buffer_length_before)
          buffer_index = read_offset-buffer_start
#          print("ADV:")
#          print("  buffer {}".format(buffer))
#          print("  buffer_start {}".format(buffer_start))
#          print("  buffer_index {}".format(buffer_index))

      # get the correct item from the buffer
      if buffer_index<0:
        raise Exception("chunkiter.tools.tee buffer exhausted")

      if buffer_index>=len(buffer):
        return
      else:
        yield buffer[buffer_index]

      next_offsets[i] += 1

  def yield_generators():
    for i in itertools.count():
      yield gen(i)

  generator_of_generators = yield_generators()

  if n is None:
    return generator_of_generators
  else:
    return tuple(next(generator_of_generators) for i in range(n))

class ReusableGenerator:
  """Wrap an iterator so it can be iterated multiple times.

  Uses :func:`tee` to create fresh iterators on demand while keeping
  memory bounded.

  Args:
      generator: An iterator or generator.

  Example:
      >>> chunks = [np.array([1., 2.]), np.array([3., 4.])]
      >>> reusable = chunkiter.ReusableGenerator(chunks)
      >>> list(reusable)
      [array([1., 2.]), array([3., 4.])]
      >>> list(reusable)  # can iterate again
      [array([1., 2.]), array([3., 4.])]
  """
  def __init__(self, generator):
    self.tee = tee(generator, n=None)
    self.main = next(self.tee)

  def __iter__(self):
    return next(self.tee)

  def __next__(self):
    return next(self.main)

def split(iterator):
  """Split a tuple-chunk iterator into individual per-entry iterators.

  Args:
      iterator: Iterator yielding tuples of np.ndarray chunks.

  Returns:
      tuple of iterators: One iterator per entry in the tuple.

  Example:
      >>> import numpy as np
      >>> chunks = [(np.array([1.]), np.array([7.])),
      ...           (np.array([2.]), np.array([8.]))]
      >>> data, labels = chunkiter.split(chunks)
      >>> list(data)
      [array([1.]), array([2.])]
      >>> list(labels)
      [array([7.]), array([8.])]
  """
  first,iterator = peek(iterator)
  n = len(first)
  subiterators = tee(iterator, n)
  subiterators = tuple((lambda i: (chunk[i] for chunk in subiterators[i]))(j) for j in range(n))
  return subiterators

def linspace(start, stop, points, chunksize, endpoint=True):
  """Like ``np.linspace`` but yields chunks instead of a single array.

  Useful for generating large coordinate arrays without holding them all
  in memory.

  Args:
      start (float): Start value.
      stop (float): End value.
      points (int): Total number of points.
      chunksize (int): Points per yielded chunk.
      endpoint (bool): If ``True``, *stop* is the last point.

  Yields:
      np.ndarray: Chunks of the linearly spaced values.

  Example:
      >>> list(chunkiter.linspace(0, 1, 5, 3))
      [array([0.  , 0.25, 0.5 ]), array([0.75, 1.  ])]
  """
  if endpoint: diff = (stop-start)/(points-1)
  else: diff = (stop-start)/points

  yielded_points = 0
  while yielded_points<points:
    chunk = np.arange(min(chunksize,points-yielded_points))*diff + start
    yield chunk
    yielded_points += chunk.size
    start = chunk[-1] + diff

def sosfilt(sos, iterator):
  """Apply an IIR filter (second-order sections) to a chunk iterator.

  Maintains filter state across chunk boundaries.

  Args:
      sos (np.ndarray): Second-order sections coefficient array.
      iterator: Iterator yielding np.ndarray chunks.

  Yields:
      np.ndarray: Filtered chunks.

  Example:
      >>> from scipy.signal import butter
      >>> sos = butter(4, 0.1, output="sos")
      >>> chunks = [np.random.default_rng(0).standard_normal(100)]
      >>> for filtered in chunkiter.sosfilt(sos, chunks):
      ...     print(filtered.shape)
      (100,)
  """
  z = np.zeros((sos.shape[0], 2))
  for chunk in iterator:
    output_chunk, z = scipy.signal.sosfilt(sos, chunk, zi=z)
    yield output_chunk

def sosfiltfilt(sos, iterator):
  """Apply a zero-phase IIR filter (forward-backward) to a chunk iterator.

  Filters the data forward, caches it, then filters the reversed result
  backward to achieve zero phase.

  Args:
      sos (np.ndarray): Second-order sections coefficient array.
      iterator: Iterator yielding np.ndarray chunks (ideally with an
          ``identifier`` attribute for caching).

  Returns:
      iterable: Zero-phase filtered chunk iterable.

  Example:
      >>> from scipy.signal import butter
      >>> sos = butter(4, 0.1, output="sos")
      >>> source = chunkiter.IterableH5Chunks("data.h5", "data")
      >>> filtered = chunkiter.sosfiltfilt(sos, source)
      >>> chunkiter.chunks_to_h5(filtered, "filtered.h5")
  """
  identifier = ("sosfiltfilt","filt1",iterator.identifier) if hasattr(iterator, "identifier") else ()
  filt1 = cache(sosfilt(sos, iterator), *identifier)
  identifier = ("sosfiltfilt","filt2",iterator.identifier) if hasattr(iterator, "identifier") else ()
  filt2 = reversed(cache(sosfilt(sos,reversed(filt1)), *identifier))
  return filt2

def peek(iterator, N=None):
  """Peek at the first item(s) of an iterator without consuming them.

  Returns the peeked data and a fresh iterator that yields everything
  (including the peeked items).

  Args:
      iterator: Source iterator.
      N (int, optional): Number of items to peek.  If ``None``, peeks one item.

  Returns:
      tuple: ``(peeked, rest)`` — the peeked item(s) and a fresh iterator
      over all data.

  Example:
      >>> chunks = iter([np.array([1., 2.]), np.array([3., 4.])])
      >>> first, rest = chunkiter.peek(chunks)
      >>> first
      array([1., 2.])
      >>> list(rest)
      [array([1., 2.]), array([3., 4.])]
  """
  peeker, items = tee(iter(iterator), max_buffer=1 if N is None else N)
  return (next(peeker) if N is None else [next(peeker) for i in range(N)]), items

def head(iterator, N=None):
  """Extract the first *N* samples from a chunk iterator.

  Returns the concatenated first *N* samples and a fresh iterator that
  continues from where it left off (including leftover partial chunk data).

  Args:
      iterator: Iterator yielding np.ndarray chunks.
      N (int): Number of samples to extract along axis 0.

  Returns:
      (np.ndarray, iterator): Extracted samples and the continuation iterator.

  Example:
      >>> chunks = [np.array([1., 2., 3.]), np.array([4., 5., 6.])]
      >>> head, rest = chunkiter.head(iter(chunks), 4)
      >>> head
      array([1., 2., 3., 4.])
      >>> chunkiter.concatenate(rest)
      array([5., 6.])
  """
  peeked = []
  n = 0
  while n<N:
    chunk = next(iterator)
    peeked.append(chunk)
    n += chunk.shape[0]

  def _():
    yield from peeked
    yield from iterator

  return concatenate(peeked)[:N,...], _()

def start_after(chunks, n, chunk_size="same"):
  """Drop the first *n* samples from a chunk iterator.

  Args:
      chunks: Iterator yielding np.ndarray chunks.
      n (int): Number of samples to drop from the beginning.
      chunk_size ("same", "const", or int): Output chunk sizing.
          ``"same"`` (default) preserves input chunk boundaries,
          merely trimming the leading chunk.
          ``"const"`` rechunks so every output chunk has the
          same size as the first input chunk.
          An integer value is passed directly to
          :func:`chunkiter.rechunk`.

  Yields:
      np.ndarray: Chunks with the first *n* samples removed.

  Example:
      >>> chunks = iter([np.array([1., 2., 3., 4., 5.])])
      >>> list(_drop_transient(chunks, 2))
      [array([3., 4., 5.])]
  """
  def _trim(src):
    dropped = 0
    for chunk in src:
      if dropped >= n:
        yield chunk
      elif dropped + len(chunk) > n:
        yield chunk[n - dropped:]
        dropped = n
      else:
        dropped += len(chunk)

  if chunk_size == "same":
    yield from _trim(chunks)
    return

  chunks, peek = itertools.tee(chunks)
  try:
    first = next(peek)
  except StopIteration:
    return
  cs = first.shape[0] if chunk_size == "const" else chunk_size

  yield from rechunk(_trim(chunks), cs)


def stop_after(iterator, N):
  """Yield only the first *N* samples from a chunk iterator.

  Args:
      iterator: Iterator yielding np.ndarray chunks.
      N (int): Maximum number of samples to yield along axis 0.

  Yields:
      np.ndarray: Chunks trimmed to *N* total samples.

  Example:
      >>> chunks = iter([np.array([1., 2., 3.]), np.array([4., 5., 6.])])
      >>> list(chunkiter.stop_after(chunks, 4))
      [array([1., 2., 3.]), array([4.])]
  """
  while N>0:
    chunk = next(iterator)
    r = chunk[:N,...]
    yield r
    N = N - r.shape[0]

def cumsum(iterator, initial=0):
  """Cumulative sum along the first dimension of a chunk iterator.

  Maintains the running total across chunk boundaries.

  Args:
      iterator: Iterator yielding np.ndarray chunks.
      initial (float or np.ndarray): Starting value for the cumulative sum.

  Yields:
      np.ndarray: Cumulative sum chunks.

  Example:
      >>> chunks = [np.array([1., 2.]), np.array([3., 4.])]
      >>> list(chunkiter.cumsum(chunks))
      [array([1., 3.]), array([6., 10.])]
  """
  for chunk in iterator:
    cumulative = np.cumsum(chunk, axis=0) + initial
    yield cumulative
    initial = cumulative[-1,...]

def add_brownian_noise(iterator, rms, frq_min, frq_max, order=50):
  """Add band-limited Brownian noise to a chunk iterator.

  Generates Brownian noise by integrating white noise, then band-pass
  filters it to the given frequency range.  The noise has the specified RMS.

  Args:
      iterator: Iterator yielding np.ndarray chunks.
      rms (float): Desired RMS amplitude of the noise.
      frq_min (float): Lower cutoff frequency (relative to Nyquist).
      frq_max (float): Upper cutoff frequency (relative to Nyquist).
      order (int): Butterworth filter order.

  Yields:
      np.ndarray: Chunks with Brownian noise added.

  Example:
      >>> import numpy as np
      >>> rng = np.random.default_rng(0)
      >>> chunks = [np.zeros(1000)]
      >>> noisy = chunkiter.add_brownian_noise(chunks, rms=1e-4, frq_min=1e-4, frq_max=1e-2)
      >>> list(noisy)[0].shape
      (1000,)
  """

  std = np.sqrt( rms**2*(2*np.pi)**2/2/(frq_min**-1-frq_max**-1) )

  iterator, iterator_ = itertools.tee(iterator)
  white_noise = (np.random.normal(size=chunk.size, scale=std) for chunk in iterator_)
  brownian_noise = cumsum(white_noise)

  sos = scipy.signal.butter(order, (2*frq_min, 2*frq_max), "bandpass", output="sos")
  brownian_noise_filtered = sosfilt(sos, brownian_noise)

  yield from (chunk+bn_chunk for chunk,bn_chunk in zip(iterator, brownian_noise_filtered))
