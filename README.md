A simple approach and library for doing numpy computations with larger-than-memory data.

Example
-------

Suppose you have a huge 2D `complex128` array in a HDF5 file and want to do an `fft` on it along one axis.
You could first load it with pytables and then do the fft:

```python
import numpy as np
import tables

# load array with pytables (shape 100000x8192, 13 GB)
array = tables.open_file("test.h5", "r").root["data"][...]

# do the fft
array_transformed = np.fft.fft(array, axis=1)
```

If the array does not fit into memory, this will fail. However, you can use python generator expressions nicely to go through the array in chunks of 1024x8192 (130 MB):

```python
import numpy as np
import tables
import chunkiter

# load array with pytables (shape 100000x8192, 13 GB)
dataset = tables.open_file("test.h5", "r").root["data"]
array = (dataset[s.start:s.stop,...] for s in chunkiter.sliceiter(1024, dataset.shape[0]))

# do the fft
array_transformed = (np.fft.fft(array_chunk, axis=1) for array_chunk in array)

# save result to output.h5
chunkiter.chunks_to_h5(array_transformed, "output.h5")
```

The trick here is that generator expressions are evaluated lazily, so at no time will all the data be in memory simultaneously.
The chunkiter module helps you here with two functions, `chunkiter.sliceiter` for dividing the large file, and `chunkiter.chunks_to_h5`
for saving a generator expression to an HDF5 file.

For optimum performance, the input file test.h5 should use Blosc compression and chunking.
The recommended way of using `chunkiter` is to have the chunkshape in the generator expressions (here, 1024x8192) agree with the chunkshape of the input HDF5 file.
If this is the case, you can just use `chunkiter.IterableH5Chunks` for loading the data:

```python
import numpy as np
import chunkiter

# load array (shape 100000x8192, 13 GB)
array = chunkiter.IterableH5Chunks("test.h5", "data")

# do the fft
array_transformed = (np.fft.fft(array_chunk, axis=1) for array_chunk in array)

# save result to output.h5
chunkiter.chunks_to_h5(array_transformed, "output.h5")
```

When using `chunkiter.chunks_to_h5`, the output HDF5 file also uses the same chunkshape as the saved generator expression.

Caching
-------

If you want to save intermediate results to disk (for example if you want to process the same data in two different ways, which is not possible with generator expressions because they can't be restarted), you can use chunkiter.cache like this:

```python
import numpy as np
import chunkiter

# load array (shape 100000x8192, 13 GB)
array = chunkiter.IterableH5Chunks("test.h5", "data")

# do the fft
array_transformed = chunkiter.cache((np.fft.fft(array_chunk, axis=1) for array_chunk in array), "fft(test.h5)")

# do other stuff
array_transformed_half = (array_transformed_chunk/2 for array_transformed_chunk in array_transformed)
array_transformed_double = (array_transformed_chunk*2 for array_transformed_chunk in array_transformed)

# save result to output.h5
chunkiter.chunks_to_h5(array_transformed_half, "output1.h5")
chunkiter.chunks_to_h5(array_transformed_double, "output2.h5")
```

The computation is only performed the first time the script is run. After that, the cached result is used. The `"fft(test.h5)"` should be a unique identifier for the generator expression that you pass to `chunkiter.cache` -  if it changes, a new computation is done. You can also specify multiple identifiers, for example one for the operation that the generator expression does and one for the input data the generator expression processes.
For your convenience, the results of `chunkiter.IterableH5Chunks` and `chunkiter.cache` have identity attributes that you can use for that, e.g.:

```python
import numpy as np
import chunkiter

# load array (shape 100000x8192, 13 GB)
array = chunkiter.IterableH5Chunks("test.h5", "data")

# do the fft
array_transformed = chunkiter.cache((np.fft.fft(array_chunk, axis=1) for array_chunk in array), "fft", array.identity)

# do other stuff
array_transformed_half = chunkiter.cache((array_transformed_chunk/2 for array_transformed_chunk in array_transformed), "half", array_transformed.identity)
array_transformed_double = chunkiter.cache((array_transformed_chunk*2 for array_transformed_chunk in array_transformed), "double", array_transformed.identity)

# save result to output.h5
chunkiter.chunks_to_h5(array_transformed_half, "output1.h5")
chunkiter.chunks_to_h5(array_transformed_double, "output2.h5")
```
