import numpy as np

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
