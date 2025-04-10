import os, itertools, traceback, hashlib, time, collections, uuid, tempfile

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
def cache(iterator, *identifiers, active=True, cachedir=None, verbose=True):
  tempdir = None
  if len(identifiers):
    identifier, *input_identifiers = identifiers
  else:
    identifier, *input_identifiers = str(uuid.uuid4()),
    if cachedir is None:
      tempdir = tempfile.TemporaryDirectory()
      cachedir = tempdir.name

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

  r = IterableH5Chunks(path)
  if tempdir is not None: r.__tempdir = tempdir # so that tempdir will not be deleted as long as r exists

  return r

def pre_rechunk(data, chunk_size, overlap_size=0):
  """
  basically, rechunking without the concatenation
  """

  if not chunk_size-overlap_size>0: raise ValueError("need chunk_size>overlap_size")

  input_chunks = [next(data)]
  input_start_i = 0
  input_stop_i = input_chunks[0].shape[0]
  output_start_i = 0

  while True:
    # clean up unneeded input chunks at head
    while len(input_chunks) and output_start_i>input_start_i+input_chunks[0].shape[0]:
      input_start_i += input_chunks[0].shape[0]
      input_chunks.pop(0)

    # extend tail of input chunks as needed
    while output_start_i+chunk_size>input_stop_i:
      try:
        next_chunk = next(data)
      except StopIteration:
        break
      input_chunks.append(next_chunk)
      input_stop_i += input_chunks[-1].shape[0]

    # prepare list of views that can be concatenated
    ret = []
    skip = output_start_i-input_start_i
    N = 0
    for chunk in input_chunks:
      # skip beginning to start at output_start_i
      chunk_relevantpart = chunk[skip:,...]
      skip -= chunk.shape[0]-chunk_relevantpart.shape[0]

      # skip end to get exactly chunk size
      chunk_relevantpart = chunk_relevantpart[:chunk_size-N,...]
      N += chunk_relevantpart.shape[0]

      if chunk_relevantpart.shape[0]>0: ret.append(chunk_relevantpart)

    if N>0: yield tuple(ret)

    if N<chunk_size: break

    output_start_i += chunk_size-overlap_size

def rechunk(data, chunk_size, overlap_size=0, padding=False, concatenate=np.concatenate):
  data = pre_rechunk(data, chunk_size, overlap_size)

  for arrays_for_concatenation in data:
    if padding:
      actual_size = sum(a.shape[0] for a in arrays_for_concatenation)
      dtype = arrays_for_concatenation[-1].dtype
      shape = (chunk_size-actual_size,) + arrays_for_concatenation[-1].shape[1:]
      arrays_for_concatenation = arrays_for_concatenation + (np.zeros(shape, dtype=dtype),)

      yield actual_size, concatenate(arrays_for_concatenation)
    else:
      yield concatenate(arrays_for_concatenation)

###

def normalize_bodyfun(bodyfun):
  if getattr(bodyfun, "has_counter", False):
    bodyfun_with_counter = bodyfun
  else:
    bodyfun_with_counter = lambda chunk_i,*args,**kwargs: bodyfun(*args,**kwargs)

  if getattr(bodyfun, "has_carry", False):
    bodyfun_with_counter_and_carry = bodyfun_with_counter
  else:
    bodyfun_with_counter_and_carry = lambda chunk_i,chunk,carry=None: (bodyfun_with_counter(chunk_i,chunk), None)

  if hasattr(bodyfun, "initial_carry"):
    bodyfun_with_counter_and_carry.initial_carry = bodyfun.initial_carry

  bodyfun_with_counter_and_carry.has_counter = True
  bodyfun_with_counter_and_carry.has_carry = True

  return bodyfun_with_counter_and_carry

def apply(bodyfun, iterator, yield_carry=False):
  """
    Applies callback function `bodyfun` on each entry of a an iterator.
    The signature of the callback must be bodyfun(chunk)->chunk.
    If bodyfun.has_carry is set to True, it must be
      bodyfun(chunk,carry)->(chunk,carry),
    where the argument carry is set to the returned carry of the last iteration.
    For the first iteration, the carry argument is set to None, or, if set,
    to bodyfun.initial_carry. If bodyfun.has_counter is set to True, the
    signature of bodyfun has an additional argument in the first position, which
    passes the iteration number.
  """
  bodyfun = normalize_bodyfun(bodyfun)

  carry = None
  for chunk_i, chunk in enumerate(iterator):
    if chunk_i==0:
      if hasattr(bodyfun, "initial_carry"): chunk, carry = bodyfun(chunk_i, chunk, bodyfun.initial_carry)
      else: chunk, carry = bodyfun(chunk_i, chunk)
    else:
      chunk, carry = bodyfun(chunk_i, chunk, carry)

    yield (chunk, carry) if yield_carry else chunk

def chain(*bodyfuns):
  """
    Returns a new bodyfun for `apply` resulting from the sequential application
    of the bodyfuns passed as arguments.
  """

  bodyfuns = [normalize_bodyfun(bodyfun) for bodyfun in bodyfuns]
  initial_carry = [getattr(bodyfun, "initial_carry", None) for bodyfun in bodyfuns]

  def bodyfun(chunk_i, chunk, carry=initial_carry):
    new_carries = []
    for bodyfun,carry_ in zip(bodyfuns,carry):
      chunk, carry = bodyfun(chunk_i, chunk, carry_)
      new_carries.append(carry)

    return chunk, new_carries

  bodyfun.has_carry = True
  bodyfun.has_counter = True
  bodyfun.initial_carry = initial_carry

  return bodyfun

def per_entry(*bodyfuns):
  """
    Returns a new bodyfun for `apply` that processes tuples.
    Each entry of the tuple is processed by the bodyfuns passed as
    arguments.
  """
  bodyfuns = [normalize_bodyfun(bodyfun) for bodyfun in bodyfuns]
  initial_carry = [getattr(bodyfun, "initial_carry", None) for bodyfun in bodyfuns]

  def bodyfun(chunk_i, chunk, carry=initial_carry):
    new_chunk = []
    new_carries = []
    for bodyfun,chunk_,carry_ in zip(bodyfuns,chunk,carry):
      chunk__, carry__ = bodyfun(chunk_i, chunk_, carry_)
      new_chunk.append(chunk__)
      new_carries.append(carry__)

    return tuple(new_chunk), tuple(new_carries)

  bodyfun.has_carry = True
  bodyfun.has_counter = True
  bodyfun.initial_carry = initial_carry

  return bodyfun
