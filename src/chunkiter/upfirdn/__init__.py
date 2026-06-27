"""
streaming upfirdn
"""

import itertools
import numpy as np

__all__ = ['upfirdn']

def upfirdn_init(h, up=1, down=1, np=np):
    h = np.asarray(h)
    
    # Pad h to a multiple of up for polyphase decomposition
    pad_len = (up - len(h) % up) % up
    if pad_len > 0:
        h_pad = np.concatenate([h, np.zeros(pad_len, dtype=h.dtype)])
    else:
        h_pad = h
        
    h_poly = h_pad.reshape(-1, up).T
    K = h_poly.shape[1]

    return {
        "h": h,
        "h_poly": h_poly,
        "up": int(up),
        "down": int(down),
        "buf": np.zeros(K - 1, dtype=h.dtype),
        "n": 0,
        "np": np,
        "withheld_y": np.zeros(0, dtype=h.dtype),
    }

def upfirdn_chunk(x, state):
    np = state["np"]

    h_poly = state["h_poly"]
    up = state["up"]
    down = state["down"]
    buf = state["buf"]
    n = state["n"]
    withheld_y = state["withheld_y"]
    K = h_poly.shape[1]

    x = np.asarray(x)

    if len(x) == 0:
        return np.zeros(0, dtype=x.dtype), state

    # prepend history buffer
    if K > 1:
        xin = np.concatenate([buf, x])
    else:
        xin = x

    outs_p = []
    # FIR filter each phase
    if K == 1:
        for p in range(up):
            outs_p.append(xin * h_poly[p, 0])
    else:
        for p in range(up):
            outs_p.append(np.convolve(xin, h_poly[p], mode='valid'))

    # Interleave
    y_chunk = np.stack(outs_p, axis=1).reshape(-1)

    if len(withheld_y) > 0:
        y_chunk = np.concatenate([withheld_y, y_chunk])

    if up > 1:
        withheld_y = y_chunk[-(up - 1):]
        y_chunk = y_chunk[:-(up - 1)]

    # Downsampling
    start = (-n) % down
    out = y_chunk[start::down]

    # update buffer
    n += len(y_chunk)
    if K > 1:
        buf = xin[-(K - 1):]

    return np.asarray(out, dtype=x.dtype), {
        **state,
        "buf": buf,
        "n": n,
        "withheld_y": withheld_y,
    }

def upfirdn_flush(state):
    np = state["np"]

    h = state["h"]
    h_poly = state["h_poly"]
    up = state["up"]
    down = state["down"]
    buf = state["buf"]
    n = state["n"]
    withheld_y = state["withheld_y"]
    K = h_poly.shape[1]

    if K > 1:
        xpad = np.concatenate([buf, np.zeros(K - 1, dtype=buf.dtype)])
    else:
        xpad = np.zeros(0, dtype=buf.dtype)

    outs_p = []
    if K == 1:
        for p in range(up):
            outs_p.append(xpad * h_poly[p, 0])
    else:
        for p in range(up):
            outs_p.append(np.convolve(xpad, h_poly[p], mode='valid'))

    if len(outs_p) > 0 and len(outs_p[0]) > 0:
        y_flush = np.stack(outs_p, axis=1).reshape(-1)
    else:
        y_flush = np.zeros(0, dtype=h.dtype)

    if len(withheld_y) > 0:
        y_flush = np.concatenate([withheld_y, y_flush])

    # match Scipy flush length exactly by truncating
    if len(h) - 1 > 0:
        y_flush = y_flush[:len(h) - 1]
    else:
        y_flush = np.zeros(0, dtype=h.dtype)

    if len(y_flush) == 0:
        return np.zeros(0, dtype=h.dtype)

    start = (-n) % down
    out = y_flush[start::down]

    return np.asarray(out, dtype=h.dtype)

def upfirdn(h, x_stream, up=1, down=1, flush=True, np=np):
    """
    Streaming upfirdn.

    Parameters
    ----------
    h : FIR filter
    x_stream : generator yielding ndarray chunks
    up : int
    down : int
    flush : bool
        If True, emit tail samples like scipy/matlab.
    np : module (numpy or jax.numpy)

    Yields
    ------
    ndarray chunks
    """

    state = upfirdn_init(h, up=up, down=down, np=np)

    for chunk in x_stream:
        y, state = upfirdn_chunk(chunk, state)
        if y.size:
            yield y

    if flush:
        y_tail = upfirdn_flush(state)
        if y_tail.size:
            yield y_tail

def test_upfirdn_against_scipy():
    import numpy as np
    import scipy.signal
    import time

    rng = np.random.default_rng(0)

    h = np.array([0.2, 0.5, 0.3])
    x = rng.standard_normal(10000000)

    # split into random chunks
    cuts = np.sort(rng.integers(1, len(x), size=10))
    chunks = np.split(x, cuts)

    start_time = time.time()
    y_stream = np.concatenate(list(upfirdn(h, iter(chunks), up=2, down=3, flush=True)))
    np_time = time.time() - start_time

    # scipy reference
    start_time = time.time()
    y_ref = scipy.signal.upfirdn(h, x, up=2, down=3)
    scipy_time = time.time() - start_time

    print(f"NumPy streaming time: {np_time:.4f}s")
    print(f"SciPy reference time: {scipy_time:.4f}s")

    assert np.allclose(y_stream, y_ref, atol=1e-10, rtol=1e-10)

def test_jax_upfirdn_against_scipy():
    import numpy as np
    import scipy.signal
    import jax.numpy as jnp
    import time

    rng = np.random.default_rng(0)

    h = rng.standard_normal(7).astype(np.float32)
    x = rng.standard_normal(10000000).astype(np.float32)

    # split into fixed chunks for JAX to avoid XLA recompilation overhead for every different shape
    chunks = np.split(x, 10)

    def gen_chunks():
        for c in chunks:
            yield c

    y_stream_np = np.concatenate(
        list(upfirdn(h, gen_chunks(), up=2, down=3, flush=True))
    )

    y_ref = scipy.signal.upfirdn(h, x, up=2, down=3)

    assert np.allclose(y_stream_np, y_ref, atol=1e-6, rtol=1e-6)

    chunks_jax = [jnp.asarray(c) for c in chunks]

    def gen_chunks_jax():
        for c in chunks_jax:
            yield c

    # forcing CPU platform for consistent CI/testing if GPU convolve has issues locally
    stream_jax = upfirdn(h, gen_chunks_jax(), up=2, down=3, flush=True, np=jnp)

    start_init = time.time()
    y_first = next(stream_jax)
    y_first.block_until_ready()
    jax_init_time = time.time() - start_init

    start_rest = time.time()
    y_rest = list(stream_jax)
    y_stream_jax = jnp.concatenate([y_first] + y_rest)
    y_stream_jax.block_until_ready()
    jax_rest_time = time.time() - start_rest

    print(f"JAX streaming time (initialization/first chunk): {jax_init_time:.4f}s")
    print(f"JAX streaming time (remaining chunks): {jax_rest_time:.4f}s")

    assert np.allclose(
        np.asarray(y_stream_jax),
        y_ref,
        atol=1e-5,
        rtol=1e-5,
    )

    print("✔ JAX streaming upfirdn matches SciPy reference")

if __name__ == "__main__":
    # Standard tests
    test_upfirdn_against_scipy()
    print("test_upfirdn_against_scipy passed")

    test_jax_upfirdn_against_scipy()
    print("test_jax_upfirdn_against_scipy passed")

    print("\nAll tests passed.")