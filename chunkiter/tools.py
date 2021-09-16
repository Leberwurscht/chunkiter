import itertools

import numpy as np

from .functions import rechunk

def mean(iterator):
  # np.mean along first dimension of a chunk iterator
  s = 0
  n = 0
  for d in iterator:
    s += np.sum(d,axis=0)
    n += d.shape[0]
  return s/n

def sum(iterator):
  # np.sum along first dimension of a chunk iterator
  s = 0
  for d in iterator:
    s += np.sum(daxis=0)
  return s

def unwrap(iterator):
  # np.unwrap along first dimension of a chunk iterator
  switchover = None

  for d in iterator:
    if switchover is None:
      switchover = np.zeros_like(d[:2,...])
      switchover[0,...] = d[0,...]

    switchover[1,...] = d[0,...]
    switchover = np.unwrap(switchover, axis=0)

    d[0,...] = switchover[1,...]

    d = np.unwrap(d, axis=0)
    yield d

    switchover[0,...] = d[-1,...]

def concatenate(iterator):
  # convert a chunk iterator to a conventional numpy array by concatenating
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
  iterator = iter(iterator)

  if chunksize is None:
    first = next(iterator)
    chunksize = first.shape[0]
    iterator = itertools.chain([first], iterator)

  yield from rechunk(_batchavg(iterator, batchsize, allow_remainder), chunksize)
