import os, itertools, traceback, hashlib, time, collections

import numpy as np
import tables

def sliceiter(n, stop):
  i = 0
  while i<stop:
    yield slice(i,i+n)
    i+= n

def multihash(*args, binary=False):
  h = hashlib.sha256()
  for a in args:
    h_ = hashlib.sha256()
    h_.update(a.encode("utf-8") if type(a)==str else a)
    h.update(h_.digest())

  if binary: return h.digest()
  else: return h.hexdigest()

class IterableH5Chunks(object):
  def __init__(self, filename, name=None, chunksize=None, reverse=False):
    self.filename = filename
    self.identifier = multihash(filename, str(type(name)), str(name), str(chunksize), str(reverse))
    self.chunksize = chunksize
    self.reverse = reverse

    datafile = tables.open_file(self.filename, "r")

    if name is None:
      if "data" in datafile.root: name = "data"
      else:
        name = []
        i = 0
        while True:
          name_ = "data{}".format(i)
          if name_ in datafile.root: name.append(name_)
          else: break
          i += 1
    self.name = name

    if type(self.name) in [list, tuple]:
      self.shape = []
      self.size = []
      self.chunksize = []
      if chunksize is not None: raise NotImplementedError("chunksize!=None not supported when yielding tuples")
      for name_ in self.name:
        array = datafile.root[name_]
        self.shape.append( array.shape )
        self.size.append( int(np.prod(array.shape)) )
        self.chunksize.append( array.chunkshape[0] )
    else:
      array = datafile.root[self.name]
      self.shape = array.shape
      self.size = int(np.prod(self.shape))
      self.chunksize = chunksize if chunksize is not None else array.chunkshape[0]

    datafile.close()

  def __iter__(self):
    datafile = tables.open_file(self.filename, "r")

    if type(self.name) in [list, tuple]: # yielding tuples of arrays
      arrays = [datafile.root[name] for name in self.name]

      if self.reverse:
        startindices_iterators = [reversed(range(0, shape[0], chunksize)) for shape, chunksize in zip(self.shape, self.chunksize)]
        for startindices in zip(*startindices_iterators):
          yield tuple(array[startindex:startindex+chunksize,...][::-1,...] for array, startindex, chunksize in zip(arrays, startindices, self.chunksize))
      else:
        startindices_iterators = [range(0, shape[0], chunksize) for shape, chunksize in zip(self.shape, self.chunksize)]
        for startindices in zip(*startindices_iterators):
          yield tuple(array[startindex:startindex+chunksize,...] for array, startindex, chunksize in zip(arrays, startindices, self.chunksize))
      
    else: # yielding single arrays
      array = datafile.root[self.name]

      if self.reverse:
        for startindex in reversed(range(0, self.shape[0], self.chunksize)):
          yield array[startindex:startindex+self.chunksize,...][::-1,...]
      else:
        for startindex in range(0, self.shape[0], self.chunksize):
          yield array[startindex:startindex+self.chunksize,...]

    datafile.close()

  def __reversed__(self):
    return IterableH5Chunks(self.filename, self.name, self.chunksize, not self.reverse)

class IdentifierIterator(object):
  def __init__(self, iterator, *identifiers):
    self.iterator = iterator
    self.identifier = multihash(*identifiers)

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.iterator)

def yielding_chunks_to_h5(iterator, filename, name=None, expectedchunks=128, verbose=False, preprocessor=None, skip=1):
  filters = tables.Filters(complevel=5, complib='blosc:lz4')

  filenames = filename if type(filename)==tuple else (filename,)
  names = name if type(name)==tuple else (name,)
  # TO DO: check if all cases are properly handled
  #   v1,   | fn      | None  | fn.data
  #   v1,v2 | fn      | None  | fn.data0 fn.data1
  #   v1,v2 | fn1,fn2 | None  | fn1.data0 fn2.data0
  #   v1,v2 | fn1,fn2 | n1,n2 | fn1.name1 fn2.name2
  #   v1,v2 | fn      | n1,n2 | fn.name1 fn.name2
  #   v1,v2 | fn1,fn2 | n     | fn1.name fn2.name
  #   v1,v2 | fn      | n     | error

  unnamed_counter = collections.Counter()
  datafiles = []
  datasets = []
  nottuple = False

  for chunk_i,data in enumerate(iterator):
    data_original = data
    if preprocessor is not None: data = preprocessor(data)

    if not type(data)==tuple:
      data = (data,)
      nottuple = True

    if len(filenames)==1 and len(data)>1: filenames = filenames*len(data)
    if len(names)==1 and len(data)>1: names = names*len(data)

    if chunk_i%skip==0:
      if verbose: print("* ...writing chunk {}".format(chunk_i), end="\r")

      # initialize datasets
      if not len(datasets):
        occupied = []
        for fn,n,v in zip(filenames, names, data):
          if n is None:
            n = "data" if nottuple else "data{}".format(unnamed_counter[fn])
            unnamed_counter[fn] += 1

          if (fn,n) in occupied: raise ValueError("conflict: tried to write twice to dataset {} in filename {}".format(n,fn))
          occupied.append((fn,n))

          datafile = tables.open_file(fn, "a")
          datafiles.append(datafile)
          if n in datafile.root: raise IOError("{} in {} already contains data".format(n, fn))
          atom = tables.Atom.from_dtype(v.dtype)
          shape = (0,)+v.shape[1:]
          datafile.create_earray(datafile.root, n, atom=atom, shape=shape, chunkshape=v.shape, expectedrows=expectedchunks*v.shape[0], filters=filters)
          datasets.append(datafile.root[n])

      for d,v in zip(datasets, data):
        d.append(v)

    yield data_original

  if verbose: print()

  for datafile in datafiles: datafile.close()

def chunks_to_h5(*args, **kwargs):
  for i in yielding_chunks_to_h5(*args, **kwargs): pass

def array_to_h5(filename, name, data):
  datafile = tables.open_file(filename, "a")
  if name not in datafile.root: datafile.create_array(datafile.root, name, atom=tables.Atom.from_dtype(data.dtype), shape=data.shape)
  datafile.root[name][...] = data
  datafile.close()

def array_from_h5(filename, name):
  datafile = tables.open_file(filename, "r")
  data = datafile.root[name][...]
  datafile.close()
  return data

default_cachedir = "cache"
def cache(iterator, identifier, *input_identifiers, active=True, cachedir=None, verbose=True):
  if type(identifier)==tuple: identifier, version = identifier
  else: version = "0"

  if len(input_identifiers):
    input_identifier = multihash(*input_identifiers)
  else:
    input_identifier = "0"

  filename = identifier+"."+version+"."+input_identifier+".h5"
  cachedir = cachedir if cachedir is not None else default_cachedir
  path = os.path.join(cachedir, filename)

  # handle active=False - just pass through the iterator, but with an identifier attached
  if not active:
    iterator = iter(iterator) # just in case iterator is a list
    return IdentifierIterator(iterator, path)

  # TO DO: clean up files with same identifier but different version

  # in case we have complete cached data, return it
  have_data = False
  if os.path.exists(path):
    try: have_data = array_from_h5(path, "_finished")
    except: pass

  if have_data:
    if verbose: print("using {}.".format(path))
    return IterableH5Chunks(path)

  # otherwise, first compute it
  if os.path.exists(path): os.remove(path)

  if verbose:
    tb = "".join(traceback.format_stack()[:-1]).split("\n")
    tb = "\n".join(["*"+line[1:80] for line in tb])

    print()
    print("*"*80)
    print("* chunkiter.cache called from\n*")
    print(tb)
    print("* saving to {}.".format(path))

  os.makedirs(cachedir, exist_ok=True)
  t_start = time.time()

  chunks_to_h5(iterator, path, verbose=verbose)

  t_total = time.time() - t_start
  array_to_h5(path, "_computation_time", np.array([t_total]))
  array_to_h5(path, "_finished", np.array([True]))
  if verbose:
    print("* done.")
    print("*"*80)
    print()

  return IterableH5Chunks(path)

def rechunk(iterator, chunksize):
  current_chunk = None
  start_index = 0
  for d in iterator:
    if current_chunk is None: current_chunk = np.empty((chunksize,)+d.shape[1:], dtype=d.dtype)

    while True:
      if start_index+d.shape[0]>=chunksize:
        # chunk to be yielded gets full
        current_chunk[start_index:] = d[:chunksize-start_index,...]
        d = d[chunksize-start_index:,...]
        yield current_chunk
        current_chunk = np.empty((chunksize,)+d.shape[1:], dtype=d.dtype)
        start_index = 0
      else:
        # chunk to be yielded does not get full
        current_chunk[start_index:start_index+d.shape[0]] = d
        start_index += d.shape[0]
        break

  if start_index!=0: yield current_chunk[:start_index]
