import os, itertools, traceback, hashlib, time, collections, uuid, tempfile

import numpy as np
import tables

def sliceiter(n, stop):
  """Yield ``slice`` objects dividing ``range(stop)`` into chunks of size *n*.

  Useful for slicing large HDF5 datasets chunk-by-chunk::

      dataset = tables.open_file("data.h5").root.data
      chunks = (dataset[s, ...] for s in chunkiter.sliceiter(1024, dataset.shape[0]))

  Args:
      n (int): Chunk size.
      stop (int): Upper bound of the range (exclusive).

  Yields:
      slice: Slice ``slice(i, i+n)`` for each chunk.

  Example:
      >>> list(chunkiter.sliceiter(3, 7))
      [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
  """
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
  """Iterable that reads an HDF5 dataset chunk-by-chunk without loading all data into memory.

  Matches the native chunk shape of the HDF5 file by default for optimal
  I/O performance.  Supports iterating multiple datasets simultaneously
  (yielding tuples) and reverse iteration.

  Each instance has an ``identifier`` attribute for use with :func:`cache`.

  Args:
      filename (str): Path to the HDF5 file.
      name (str, list, None): Dataset name(s).  If ``None``, auto-discovers
          ``data`` or ``data0``, ``data1``, ... datasets.
      chunksize (int, None): Override chunk size.  ``None`` uses the file's
          native chunk size.
      reverse (bool): If ``True``, iterate in reverse order.

  Yields:
      np.ndarray or tuple of np.ndarray: Data chunks.

  Example:
      >>> array = chunkiter.IterableH5Chunks("test.h5", "data")
      >>> for chunk in array:
      ...     print(chunk.shape)
      >>> print(array.identifier)
  """
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

class IterableBinaryFileChunks(object):
  """Iterator reading from binary format (streaming-capable, also over sockets using ``socket.makefile``).

  Reads the format written by :func:`yielding_chunks_to_binaryfile`.
  Each instance has an ``identifier`` attribute.

  Args:
      file: A file-like object in binary read mode.
      identifier (str, optional): Identifier hash.  Auto-generated if not given.

  Yields:
      np.ndarray or tuple of np.ndarray: Deserialized chunk(s).

  Example:
      >>> import socket
      >>> sock = socket.create_connection(("localhost", 12345))
      >>> chunks = chunkiter.IterableBinaryFileChunks(sock.makefile("rb"))
      >>> for chunk in chunks:
      ...     print(chunk.shape)
  """

  def __init__(self, file, identifier=None):
    self.file = file
    self.identifier = identifier if identifier is not None else str(uuid.uuid4())

  def __iter__(self):
    while True:
      assert self.file.read(5)==b'TUPLE'

      tuple_len = np.empty(1, np.int64)
      self.file.readinto(tuple_len.view("b").data)

      arrays = []
      for i in range(tuple_len.item()):
        arrays.append(deserialize_ndarray(self.file))

      arrays = tuple(arrays)

      yield arrays[0] if len(arrays)==1 else arrays

class IdentifierIterator(object):
  """Iterator wrapper that attaches an ``identifier`` hash attribute.

  Used internally by :func:`cache` when ``active=False``.

  Args:
      iterator: Any iterator.
      *identifiers: Values to hash into the identifier.

  Example:
      >>> it = chunkiter.IdentifierIterator(range(3), "myid")
      >>> it.identifier
      '...'
      >>> list(it)
      [0, 1, 2]
  """
  def __init__(self, iterator, *identifiers):
    self.iterator = iterator
    self.identifier = multihash(*identifiers)

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.iterator)

def yielding_chunks_to_h5(iterator, filename, name=None, expectedchunks=128, verbose=False, preprocessor=None, skip=1):
  """Stream chunks from an iterator to HDF5, yielding data unchanged for further processing.

  This is the streaming variant — each chunk is written and yielded immediately.
  Use :func:`chunks_to_h5` for the fire-and-forget variant that consumes the
  entire iterator without yielding.

  Handles single arrays and tuples of arrays, writing to one or multiple files.

  Args:
      iterator: Iterator yielding ``np.ndarray`` chunks (or tuples thereof).
      filename (str or tuple): Output HDF5 filename(s).
      name (str, tuple, None): Dataset name(s).  ``None`` auto-names to ``data``
          or ``data0``, ``data1``, ...
      expectedchunks (int): Expected total number of chunks (for HDF5 extent hint).
      verbose (bool): Print progress.
      preprocessor (callable, optional): Transformation applied before writing
          (yielded data is still the original).
      skip (int): Only write/save every ``skip``-th chunk (1 = every chunk).

  Yields:
      np.ndarray or tuple of np.ndarray: Original (unprocessed) chunks.

  Example:
      >>> source = chunkiter.IterableH5Chunks("input.h5", "data")
      >>> processed = (chunk * 2 for chunk in source)
      >>> # Write to disk while still being able to further process:
      >>> saved = chunkiter.yielding_chunks_to_h5(processed, "output.h5")
      >>> chunkiter.chunks_to_binaryfile(saved, open("output.bin", "wb"))
  """
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
  """Consume an iterator and write all chunks to HDF5 (no yielding).

  Convenience wrapper around :func:`yielding_chunks_to_h5` that drains the
  iterator entirely.  Same parameters.

  Example:
      >>> source = chunkiter.IterableH5Chunks("input.h5", "data")
      >>> processed = (chunk * 2 for chunk in source)
      >>> chunkiter.chunks_to_h5(processed, "output.h5")
  """
  for i in yielding_chunks_to_h5(*args, **kwargs): pass

def array_to_h5(filename, name, data):
  """Write a single numpy array to an HDF5 file.

  Args:
      filename (str): Path to the HDF5 file.
      name (str): Dataset name inside the file.
      data (np.ndarray): Array to write.

  Example:
      >>> chunkiter.array_to_h5("output.h5", "result", np.array([1, 2, 3]))
      >>> np.allclose(chunkiter.array_from_h5("output.h5", "result"), [1, 2, 3])
      True
  """
  datafile = tables.open_file(filename, "a")
  if name not in datafile.root: datafile.create_array(datafile.root, name, atom=tables.Atom.from_dtype(data.dtype), shape=data.shape)
  datafile.root[name][...] = data
  datafile.close()

def array_from_h5(filename, name):
  """Read a single numpy array from an HDF5 file.

  Args:
      filename (str): Path to the HDF5 file.
      name (str): Dataset name inside the file.

  Returns:
      np.ndarray: The read array.

  Example:
      >>> data = chunkiter.array_from_h5("input.h5", "metadata")
  """
  datafile = tables.open_file(filename, "r")
  data = datafile.root[name][...]
  datafile.close()
  return data

def serialize_ndarray(array, file):
  file.write(b'ARRAY')

  typestr = array.dtype.str.encode("ascii")
  file.write(np.array([len(typestr)], np.int64).view("b").data)
  file.write(typestr)

  shape = np.array(array.shape, np.int64)
  file.write(np.array([shape.size], np.int64).view("b").data)
  file.write(shape.view("b").data)

  file.write(np.ascontiguousarray(array).view("b").data)

def deserialize_ndarray(file, memory_limit=512*1024**2):
  assert file.read(5)==b'ARRAY'

  typestr_len = np.empty(1, np.int64)
  file.readinto(typestr_len.view("b").data)
  assert typestr_len<memory_limit

  typestr = np.empty(typestr_len, np.int8)
  file.readinto(typestr.view("b").data)
  typestr = bytes(typestr.view("b").data).decode("ascii")

  shape_len = np.empty(1, np.int64)
  file.readinto(shape_len.view("b").data)
  assert shape_len*8<memory_limit

  shape = np.empty(shape_len, np.int64)
  file.readinto(shape.view("b").data)

  dtype = np.dtype(typestr)
  totalsize = np.prod(shape)*dtype.itemsize
  assert totalsize<memory_limit

  array = np.empty(shape, dtype=dtype)
  file.readinto(array.view("b").data)

  return array

def yielding_chunks_to_binaryfile(iterator, file, verbose=True, preprocessor=None, skip=1):
  """Write chunks to a binary file format, yielding data for further streaming.

  Streaming-capable — can write to sockets via ``socket.makefile``.
  See :class:`IterableBinaryFileChunks` for reading back.

  The binary format (repeating) consists of:

  - string ``TUPLE``
  - 64-bit integer: number of ndarrays in this tuple
  - For each ndarray: string ``ARRAY``, 64-bit integer typestr length,
    typestring, 64-bit integer ndim, shape (ndim × 64-bit), raw array data (C order)

  Args:
      iterator: Iterator yielding ``np.ndarray`` chunks (or tuples thereof).
      file: Writable binary file-like object.
      verbose (bool): Print progress with throughput.
      preprocessor (callable, optional): Transform applied before writing
          (yielded data is still original).
      skip (int): Only write every ``skip``-th chunk (1 = every chunk).

  Yields:
      np.ndarray or tuple of np.ndarray: Original chunks, unchanged.

  Example:
      >>> source = chunkiter.IterableH5Chunks("input.h5", "data")
      >>> with open("output.bin", "wb") as f:
      ...     for chunk in chunkiter.yielding_chunks_to_binaryfile(source, f):
      ...         pass  # chunks are written to disk and can also be processed here
  """

  speed_write = np.nan
  speed_current = np.nan
  speed_avg = np.nan

  total_bytes = 0
  t_start = time.time()
  t_chunk_start = time.time()

  for data_i,data in enumerate(iterator):
    data_original = data
    if preprocessor is not None: data = preprocessor(data)

    if not type(data)==tuple:
      data = (data,)

    if data_i%skip==0:
      if verbose: print("* ...writing chunk {}, current {:.2f} MB/s, avg {:.2f} MB/s".format(data_i, speed_current/1024**2, speed_avg/1024**2), end="\r")

      file.write(b'TUPLE')
      file.write(np.array([len(data)], np.int64).view("b").data)

      chunk_bytes = 0
      for d in data:
        serialize_ndarray(d, file)
        total_bytes += d.nbytes
        chunk_bytes += d.nbytes

      now = time.time()
      speed_current = chunk_bytes/(now-t_chunk_start)
      speed_avg = total_bytes/(now-t_start)
      t_chunk_start = now

      file.flush()

    yield data_original

def chunks_to_binaryfile(*args, **kwargs):
  """Consume an iterator and write all chunks to binary format (no yielding).

  Convenience wrapper around :func:`yielding_chunks_to_binaryfile`.
  Same parameters.

  Example:
      >>> source = chunkiter.IterableH5Chunks("input.h5", "data")
      >>> with open("output.bin", "wb") as f:
      ...     chunkiter.chunks_to_binaryfile(source, f)
  """
  for i in yielding_chunks_to_binaryfile(*args, **kwargs): pass

default_cachedir = "cache"
def cache(iterator, *identifiers, active=True, cachedir=None, verbose=True):
  """Cache an iterator's output to disk for repeated consumption.

  Generator expressions can't be rewound — you can't consume the same one
  twice.  ``cache`` computes the iterator once (writing to HDF5), then returns
  an :class:`IterableH5Chunks` that can be iterated multiple times.  Subsequent
  calls with the same identifier skip computation entirely.

  The returned iterable has an ``identifier`` attribute that can be used as
  input for dependent caches.

  Args:
      iterator: Any iterator yielding ``np.ndarray`` chunks.
      *identifiers: Unique identifier(s) for the cached result.  The first
          identifier can be a ``(name, version)`` tuple for versioned caches.
          If empty, a random UUID is used with a temporary directory.
      active (bool): If ``False``, computation is skipped (pass-through with
          an :class:`IdentifierIterator` wrapper).
      cachedir (str, optional): Cache directory.  Defaults to ``"cache"``.
      verbose (bool): Print progress and cache status.

  Returns:
      :class:`IterableH5Chunks` or :class:`IdentifierIterator`: An iterable
      reading from the cached HDF5 file, or a pass-through wrapper.

  Example:
      >>> import numpy as np, chunkiter
      >>> source = chunkiter.IterableH5Chunks("data.h5", "data")
      >>> # Computation runs only the first time:
      >>> fftdata = chunkiter.cache(
      ...     (np.fft.fft(chunk, axis=1) for chunk in source),
      ...     "fft", source.identifier
      ... )
      >>> # Use result multiple times:
      >>> half = (c / 2 for c in fftdata)
      >>> double = (c * 2 for c in fftdata)
  """
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
  """Prepare chunk views for rechunking, without concatenating.

  Collects enough input chunks to form output chunks of *chunk_size*,
  yielding lists of array views ready for concatenation.  When *overlap_size*
  > 0, each output chunk overlaps the previous one by that many samples.

  Args:
      data: Iterator yielding np.ndarray chunks (any shape along axis 0).
      chunk_size (int): Desired output chunk size.
      overlap_size (int): Number of samples shared between consecutive
          output chunks.

  Yields:
      tuple of np.ndarray: Views ready for ``np.concatenate``.

  Raises:
      ValueError: If chunk_size <= overlap_size.

  Example:
      >>> import numpy as np
      >>> source = (np.ones((list(range(1, 6)))) for _ in [1])
      >>> # Actually use IterableH5Chunks or similar:
      >>> source = [np.ones((17, 3)), np.ones((17, 3)), np.ones((16, 3))]
      >>> for views in chunkiter.pre_rechunk(source, 10):
      ...     print(np.concatenate(views).shape)
      (10, 3)
      (10, 3)
      (10, 3)
      (10, 3)
      (10, 3)
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

def rechunk(data, chunk_size, overlap_size=0, padding=False, concatenate=np.concatenate, yield_remainder=True):
  """Rechunk a stream of array chunks to a different chunk size.

  Collects input chunks and repartitions them into output chunks of equal
  size *chunk_size*.  Optional overlap between consecutive chunks and
  zero-padding for the final partial chunk.

  Args:
      data: Iterator yielding np.ndarray chunks.
      chunk_size (int): Desired output chunk size along axis 0.
      overlap_size (int): Number of overlapping samples between consecutive
          output chunks.
      padding (bool): If ``True``, the last chunk is zero-padded to
          *chunk_size* and the return value is ``(actual_size, chunk)``
          instead of just ``chunk``.
      concatenate (callable): Function for concatenating array views.
          Defaults to ``np.concatenate``.
      yield_remainder (bool): If ``True`` (default) the last chunk will
          be returned even if it is smaller than the specificed *chunk_size*.

  Yields:
      np.ndarray or (int, np.ndarray): Rechunked array chunk.  With
      ``padding=True``, yields ``(actual_size, chunk)`` tuples.

  Example:
      >>> import numpy as np, chunkiter
      >>> source = [np.ones((17, 3)), np.ones((17, 3)), np.ones((16, 3))]
      >>> for chunk in chunkiter.rechunk(source, 10):
      ...     print(chunk.shape)
      (10, 3)
      (10, 3)
      (10, 3)
      (10, 3)
      (10, 3)
  """
  data = pre_rechunk(data, chunk_size, overlap_size)

  for arrays_for_concatenation in data:
    if padding:
      actual_size = sum(a.shape[0] for a in arrays_for_concatenation)
      dtype = arrays_for_concatenation[-1].dtype
      shape = (chunk_size-actual_size,) + arrays_for_concatenation[-1].shape[1:]
      arrays_for_concatenation = arrays_for_concatenation + (np.zeros(shape, dtype=dtype),)

      if actual_size==chunk_size or yield_remainder: yield actual_size, concatenate(arrays_for_concatenation)
    else:
      arr = concatenate(arrays_for_concatenation)
      if arr.shape[0]==chunk_size or yield_remainder: yield arr

###

def normalize_bodyfun(bodyfun):
  """Normalize a callable for use with :func:`apply`, :func:`chain` and :func:`per_entry`.

  Adds counter and carry support to the body function if missing.
  After normalization the callable has the signature
  ``(chunk_i, chunk, carry=None) -> (chunk, carry)`` and the attributes
  ``has_counter`` and ``has_carry`` set to ``True``.

  Args:
      bodyfun (callable): Raw body function, one of:

          - ``bodyfun(chunk) -> chunk``
          - ``bodyfun(chunk, carry) -> (chunk, carry)`` (has ``has_carry=True``)
          - ``bodyfun(chunk_i, chunk) -> chunk`` (has ``has_counter=True``)
          - ``bodyfun(chunk_i, chunk, carry) -> (chunk, carry)``

  Returns:
      callable: Normalized body function with full signature and attributes.
  """
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
  """Apply a callback to every chunk in an iterator.

  The callback can optionally maintain state across chunks via a *carry*
  value, and can receive the chunk index via a *counter*.

  The body function signature can be any of (see :func:`normalize_bodyfun`):

  * ``bodyfun(chunk) -> chunk``
  * ``bodyfun(chunk, carry) -> (chunk, carry)``  (set ``has_carry = True``)
  * ``bodyfun(chunk_i, chunk) -> chunk``  (set ``has_counter = True``)
  * ``bodyfun(chunk_i, chunk, carry) -> (chunk, carry)``

  The *carry* for the first iteration is ``bodyfun.initial_carry`` (if set)
  or ``None``.

  Args:
      bodyfun (callable): Callback (auto-normalized by :func:`normalize_bodyfun`).
      iterator: Iterator yielding chunks.
      yield_carry (bool): If ``True``, yield ``(chunk, carry)`` tuples instead
          of just chunks.

  Yields:
      np.ndarray or (np.ndarray, object): Processed chunks (optionally with carry).

  Example:
      >>> import numpy as np
      >>> # A bodyfun that accumulates a running sum:
      >>> def running_sum(chunk, carry=0):
      ...     carry = np.cumsum(chunk, axis=0) + carry
      ...     return carry, carry[-1]
      >>> running_sum.has_carry = True
      >>> running_sum.initial_carry = 0
      >>> chunks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
      >>> list(chunkiter.apply(running_sum, chunks))
      [array([1, 3, 6]), array([10, 15, 21])]
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
  """Compose multiple body functions for sequential application with :func:`apply`.

  Returns a new bodyfun that passes each chunk through *bodyfuns* in order.
  Carry values are tracked independently per body function.

  Args:
      *bodyfuns: Body functions to chain together.

  Returns:
      callable: A combined body function with ``has_carry`` and ``has_counter``
      set, suitable for :func:`apply`.

  Example:
      >>> import numpy as np
      >>> from chunkiter import chain, apply
      >>> def scale(chunk, carry):
      ...     return chunk * 2, carry
      >>> scale.has_carry = True
      >>> def add_one(chunk, carry):
      ...     return chunk + 1, carry
      >>> add_one.has_carry = True
      >>> body = chain(scale, add_one)
      >>> list(apply(body, [np.array([1, 2]), np.array([3, 4])]))
      [array([3, 5]), array([7, 9])]
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
  """Apply different body functions to each entry of tuple chunks.

  When a chunk iterator yields tuples (e.g. ``(data, labels)``), this creates
  a bodyfun that passes the *i*-th entry to the *i*-th *bodyfuns* argument.

  Args:
      *bodyfuns: One body function per tuple entry.

  Returns:
      callable: A combined body function processing tuples, with ``has_carry``
      and ``has_counter`` set.

  Example:
      >>> import numpy as np
      >>> from chunkiter import per_entry, apply
      >>> def scale(chunk, carry):
      ...     return chunk * 2, carry
      >>> scale.has_carry = True
      >>> def pass_through(chunk, carry):
      ...     return chunk, carry
      >>> pass_through.has_carry = True
      >>> body = per_entry(scale, pass_through)
      >>> chunks = [(np.array([1, 2]), np.array([7, 8]))]
      >>> list(apply(body, chunks))
      [(array([2, 4]), array([7, 8]))]
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
