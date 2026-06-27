import itertools, uuid

import numpy as np
import scipy.signal

from .functions import rechunk, cache

def mean(iterator):
  """ np.mean along first dimension of a chunk iterator """
  s = 0
  n = 0
  for d in iterator:
    s += np.sum(d,axis=0)
    n += d.shape[0]
  return s/n

def sum(iterator):
  """ np.sum along first dimension of a chunk iterator """
  s = 0
  for d in iterator:
    s += np.sum(daxis=0)
  return s

def unwrap(iterator):
  """ np.unwrap along first dimension of a chunk iterator """
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
  """ convert a chunk iterator to a conventional numpy array by concatenating """

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
  """
  """
  iterator = iter(iterator)

  if chunksize is None:
    first = next(iterator)
    chunksize = first.shape[0]
    iterator = itertools.chain([first], iterator)

  yield from rechunk(_batchavg(iterator, batchsize, allow_remainder), chunksize)

def enumerate(iterator):
  """
  """
  start = 0
  for chunk in iterator:
    n = chunk[0].shape[0] if type(chunk)==tuple else chunk.shape[0]
    counter = np.arange(n) + start
    yield counter, chunk
    start += n

def tee(iterator, n=2, max_buffer=1):
  """
  like itertools.tee, but instead of buffering all the data (and thus filling
  up RAM) until it is consumed from the returned iterators, it:
   * only keeps at most `max_buffer` items buffered
   * when returned iterators are consumed too asynchronously so that buffered
     data is not sufficient, it will raise an Exception
   * consequently, when one of the returned iterators is not consumed at all,
     this will not fill up the RAM
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
  """
  """
  def __init__(self, generator):
    self.tee = tee(generator, n=None)
    self.main = next(self.tee)

  def __iter__(self):
    return next(self.tee)

  def __next__(self):
    return next(self.main)

def split(iterator):
  """
  """
  first,iterator = peek(iterator)
  n = len(first)
  subiterators = tee(iterator, n)
  subiterators = tuple((lambda i: (chunk[i] for chunk in subiterators[i]))(j) for j in range(n))
  return subiterators

def linspace(start, stop, points, chunksize, endpoint=True):
  """
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
  """
  """
  z = np.zeros((sos.shape[0], 2))
  for chunk in iterator:
    output_chunk, z = scipy.signal.sosfilt(sos, chunk, zi=z)
    yield output_chunk

def sosfiltfilt(sos, iterator):
  """
  """
  identifier = ("sosfiltfilt","filt1",iterator.identifier) if hasattr(iterator, "identifier") else ()
  filt1 = cache(sosfilt(sos, iterator), *identifier)
  identifier = ("sosfiltfilt","filt2",iterator.identifier) if hasattr(iterator, "identifier") else ()
  filt2 = reversed(cache(sosfilt(sos,reversed(filt1)), *identifier))
  return filt2

def peek(iterator, N=None):
  """
  """
  peeker, items = tee(iter(iterator), max_buffer=1 if N is None else N)
  return (next(peeker) if N is None else [next(peeker) for i in range(N)]), items

def head(iterator, N=None):
  """
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

def stop_after(iterator, N):
  """
  """
  while N>0:
    chunk = next(iterator)
    r = chunk[:N,...]
    yield r
    N = N - r.shape[0]

def cumsum(iterator, initial=0):
  """
  """
  for chunk in iterator:
    cumulative = np.cumsum(chunk, axis=0) + initial
    yield cumulative
    initial = cumulative[-1,...]

def add_brownian_noise(iterator, rms, frq_min, frq_max, order=50):
  """
  Generate band-limited brownian noise with a certain rms
  """

  std = np.sqrt( rms**2*(2*np.pi)**2/2/(frq_min**-1-frq_max**-1) )

  iterator, iterator_ = itertools.tee(iterator)
  white_noise = (np.random.normal(size=chunk.size, scale=std) for chunk in iterator_)
  brownian_noise = cumsum(white_noise)

  sos = scipy.signal.butter(order, (2*frq_min, 2*frq_max), "bandpass", output="sos")
  brownian_noise_filtered = sosfilt(sos, brownian_noise)

  yield from (chunk+bn_chunk for chunk,bn_chunk in zip(iterator, brownian_noise_filtered))
