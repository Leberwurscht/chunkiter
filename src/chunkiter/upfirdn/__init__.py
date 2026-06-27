"""
streaming upfirdn
"""

import itertools
import numpy as np

__all__ = ['upfirdn']

def upfirdn_init(h, up=1, down=1, np=np, jit=False):
    h = np.asarray(h)
    
    # Pad h to a multiple of up for polyphase decomposition
    pad_len = (up - len(h) % up) % up
    if pad_len > 0:
        h_pad = np.concatenate([h, np.zeros(pad_len, dtype=h.dtype)])
    else:
        h_pad = h
        
    h_poly = h_pad.reshape(-1, up).T
    K = h_poly.shape[1]

    core_chunk = None
    core_flush = None

    if jit:
        try:
            import jax
            import jax.numpy as jnp
            import functools
            if np is not jnp:
                raise ValueError("jit=True is only supported when np=jax.numpy")
                
            @functools.partial(jax.jit, static_argnums=(4, 5, 6))
            def _jax_core_chunk(x, buf, withheld_y, h_poly, start, up, down):
                K = h_poly.shape[1]
                if K > 1:
                    xin = jnp.concatenate([buf, x])
                else:
                    xin = x

                outs_p = []
                if K == 1:
                    for p in range(up):
                        outs_p.append(xin * h_poly[p, 0])
                else:
                    for p in range(up):
                        outs_p.append(jnp.convolve(xin, h_poly[p], mode='valid'))

                y_chunk = jnp.stack(outs_p, axis=1).reshape(-1)

                if withheld_y.size > 0:
                    y_chunk = jnp.concatenate([withheld_y, y_chunk])

                if up > 1:
                    new_withheld_y = y_chunk[-(up - 1):]
                    y_chunk = y_chunk[:-(up - 1)]
                else:
                    new_withheld_y = withheld_y

                out = y_chunk[start::down]

                if K > 1:
                    new_buf = xin[-(K - 1):]
                else:
                    new_buf = buf

                return out, new_buf, new_withheld_y, len(y_chunk)

            @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
            def _jax_core_flush(buf, withheld_y, h_poly, start, h_len, up, down):
                K = h_poly.shape[1]
                if K > 1:
                    xpad = jnp.concatenate([buf, jnp.zeros(K - 1, dtype=buf.dtype)])
                else:
                    xpad = jnp.zeros(0, dtype=buf.dtype)

                outs_p = []
                if K == 1:
                    for p in range(up):
                        outs_p.append(xpad * h_poly[p, 0])
                else:
                    for p in range(up):
                        outs_p.append(jnp.convolve(xpad, h_poly[p], mode='valid'))

                if len(outs_p) > 0 and len(outs_p[0]) > 0:
                    y_flush = jnp.stack(outs_p, axis=1).reshape(-1)
                else:
                    y_flush = jnp.zeros(0, dtype=h_poly.dtype)

                if withheld_y.size > 0:
                    y_flush = jnp.concatenate([withheld_y, y_flush])

                if h_len - 1 > 0:
                    y_flush = y_flush[:h_len - 1]
                else:
                    y_flush = jnp.zeros(0, dtype=h_poly.dtype)

                out = y_flush[start::down]
                return out

            core_chunk = _jax_core_chunk
            core_flush = _jax_core_flush
        except ImportError:
            pass

    return {
        "h": h,
        "h_poly": h_poly,
        "up": int(up),
        "down": int(down),
        "buf": np.zeros(K - 1, dtype=h.dtype),
        "n": 0,
        "np": np,
        "withheld_y": np.zeros(0, dtype=h.dtype),
        "core_chunk": core_chunk,
        "core_flush": core_flush,
        "jit_chunk_len": None,
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
    core_chunk = state["core_chunk"]

    x = np.asarray(x)

    if len(x) == 0:
        return np.zeros(0, dtype=x.dtype), state

    start = (-n) % down

    if core_chunk is not None:
        jit_chunk_len = state["jit_chunk_len"]
        current_len = len(x)

        if jit_chunk_len is None:
            jit_chunk_len = current_len

        # Pad x if it is smaller than the expected jit_chunk_len (e.g. final chunk)
        if current_len < jit_chunk_len:
            x_padded = np.concatenate([x, np.zeros(jit_chunk_len - current_len, dtype=x.dtype)])
            out_padded, new_buf_padded, new_withheld_y_padded, y_chunk_len_padded = core_chunk(
                x_padded, buf, withheld_y, h_poly, start, up, down
            )
            
            # The exact number of theoretical output samples for the UNPADDED length.
            # y_chunk length is purely x length * up
            y_chunk_len_real = current_len * up

            # We must determine how many of these samples survived downsampling
            # We had `start = (-n) % down`
            # The indices we took were start, start+down, start+2down... strictly less than y_chunk_len_real
            # The number of elements is ceil((y_chunk_len_real - start) / down)
            if y_chunk_len_real > start:
                num_out_real = (y_chunk_len_real - start + down - 1) // down
            else:
                num_out_real = 0

            out = out_padded[:num_out_real]
            
            # History buffer: We must update the history buffer using the UNPADDED x.
            # We simply fallback to eager mode for the history buffer logic to be safe and accurate on the final chunk.
            if K > 1:
                xin_real = np.concatenate([buf, x])
                if len(xin_real) >= K - 1:
                    new_buf = xin_real[-(K - 1):]
                else:
                    new_buf = np.concatenate([np.zeros((K - 1) - len(xin_real), dtype=x.dtype), xin_real])
            else:
                new_buf = buf

            # We also must recalculate new_withheld_y mathematically
            if up > 1:
                # Eagerly compute just the last up-1 samples of y_chunk_real
                # To do this efficiently, we just use the eager fallback for the withheld tracking
                xin_real = np.concatenate([buf, x])
                outs_p = []
                if K == 1:
                    for p in range(up):
                        outs_p.append(xin_real * h_poly[p, 0])
                else:
                    for p in range(up):
                        outs_p.append(np.convolve(xin_real, h_poly[p], mode='valid'))
                y_chunk_real = np.stack(outs_p, axis=1).reshape(-1)
                if len(withheld_y) > 0:
                    y_chunk_real = np.concatenate([withheld_y, y_chunk_real])
                new_withheld_y = y_chunk_real[-(up - 1):]
            else:
                new_withheld_y = withheld_y

            n += int(y_chunk_len_real)
            return np.asarray(out, dtype=x.dtype), {
                **state,
                "buf": new_buf,
                "n": n,
                "withheld_y": new_withheld_y,
                "jit_chunk_len": jit_chunk_len,
            }

        else:
            out, new_buf, new_withheld_y, y_chunk_len = core_chunk(x, buf, withheld_y, h_poly, start, up, down)
            n += int(y_chunk_len)
            return out, {
                **state,
                "buf": new_buf,
                "n": n,
                "withheld_y": new_withheld_y,
                "jit_chunk_len": jit_chunk_len,
            }

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
    core_flush = state["core_flush"]

    start = (-n) % down

    if core_flush is not None:
        out = core_flush(buf, withheld_y, h_poly, start, len(h), up, down)
        return out

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

def upfirdn(h, x_stream, up=1, down=1, flush=True, np=np, jit=False):
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
    jit : bool
        If True and np is jax.numpy, compile core computations using jax.jit.
        Only recommended when chunk lengths are strictly uniform.

    Yields
    ------
    ndarray chunks
    """

    state = upfirdn_init(h, up=up, down=down, np=np, jit=jit)

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

    h = rng.standard_normal(7)
    x = rng.standard_normal(100000000)

    # split into equal chunks
    chunks = np.split(x, 100)
    chunk_size_mb = chunks[0].nbytes / (1024 * 1024)
    print(f"\n--- NumPy Test ---")
    print(f"Total size: {x.nbytes / (1024 * 1024):.1f} MB, Chunk size: {chunk_size_mb:.1f} MB")

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
    import jax
    import jax.numpy as jnp
    import time
    
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(0)

    h = rng.standard_normal(7)
    x = rng.standard_normal(100000000)

    # split into fixed chunks for JAX to avoid XLA recompilation overhead for every different shape
    chunks = np.split(x, 100)
    chunk_size_mb = chunks[0].nbytes / (1024 * 1024)
    print(f"\n--- JAX Test ---")
    print(f"Total size: {x.nbytes / (1024 * 1024):.1f} MB, Chunk size: {chunk_size_mb:.1f} MB")

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

    # 1. Test eager JAX streaming
    stream_jax = upfirdn(h, gen_chunks_jax(), up=2, down=3, flush=True, np=jnp, jit=False)

    start_init = time.time()
    y_first = next(stream_jax)
    y_first.block_until_ready()
    jax_init_time = time.time() - start_init

    start_rest = time.time()
    y_rest = list(stream_jax)
    y_stream_jax = jnp.concatenate([y_first] + y_rest)
    y_stream_jax.block_until_ready()
    jax_rest_time = time.time() - start_rest

    print(f"JAX (Eager) streaming time (initialization/first chunk): {jax_init_time:.4f}s")
    print(f"JAX (Eager) streaming time (remaining chunks): {jax_rest_time:.4f}s")
    print(f"JAX (Eager) streaming time (total): {jax_init_time + jax_rest_time:.4f}s")

    assert np.allclose(
        np.asarray(y_stream_jax),
        y_ref,
        atol=1e-5,
        rtol=1e-5,
    )

    # 2. Test JIT JAX streaming
    stream_jax_jit = upfirdn(h, gen_chunks_jax(), up=2, down=3, flush=True, np=jnp, jit=True)
    
    start_init_jit = time.time()
    y_first_jit = next(stream_jax_jit)
    y_first_jit.block_until_ready()
    jax_init_time_jit = time.time() - start_init_jit

    start_rest_jit = time.time()
    y_rest_jit = list(stream_jax_jit)
    y_stream_jax_jit = jnp.concatenate([y_first_jit] + y_rest_jit)
    y_stream_jax_jit.block_until_ready()
    jax_rest_time_jit = time.time() - start_rest_jit

    print(f"JAX (JIT) streaming time (compilation/first chunk): {jax_init_time_jit:.4f}s")
    print(f"JAX (JIT) streaming time (remaining chunks): {jax_rest_time_jit:.4f}s")
    print(f"JAX (JIT) streaming time (total): {jax_init_time_jit + jax_rest_time_jit:.4f}s")

    assert np.allclose(
        np.asarray(y_stream_jax_jit),
        y_ref,
        atol=1e-5,
        rtol=1e-5,
    )

    # 3. Test JIT JAX streaming with an uneven final chunk
    # This verifies our padding/trimming logic works correctly without recompilation
    uneven_cuts = np.arange(1000000, 100000000, 1000000)
    # let's make the last chunk uneven by adding a little bit of data
    x_uneven = rng.standard_normal(100000500)
    chunks_uneven = np.split(x_uneven[:100000000], 100) + [x_uneven[100000000:]]
    chunks_uneven_jax = [jnp.asarray(c) for c in chunks_uneven]

    def gen_chunks_uneven_jax():
        for c in chunks_uneven_jax:
            yield c

    stream_jax_uneven = upfirdn(h, gen_chunks_uneven_jax(), up=2, down=3, flush=True, np=jnp, jit=True)
    y_uneven_out = jnp.concatenate(list(stream_jax_uneven))
    y_uneven_ref = scipy.signal.upfirdn(h, x_uneven, up=2, down=3)

    assert np.allclose(
        np.asarray(y_uneven_out),
        y_uneven_ref,
        atol=1e-5,
        rtol=1e-5,
    )

    print("✔ JAX streaming upfirdn matches SciPy reference (including uneven final chunks)")

def test_single_chunk_performance():
    import numpy as np
    import scipy.signal
    import time
    
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        has_jax = True
    except ImportError:
        has_jax = False

    rng = np.random.default_rng(42)
    
    N = 100_000_000  # 100 Million elements
    up = 3
    down = 2
    h_len = 64
    
    print(f"\n--- Single Chunk Benchmark (N={N:,}, up={up}, down={down}, h_len={h_len}) ---")
    
    h = rng.standard_normal(h_len)
    x = rng.standard_normal(N)
    
    print(f"Input Data Size: {x.nbytes / 1024**2:.1f} MB")
    
    # --- SciPy ---
    _ = scipy.signal.upfirdn(h, x[:100], up, down) # Warmup
    t0 = time.perf_counter()
    scipy_out = scipy.signal.upfirdn(h, x, up, down)
    t1 = time.perf_counter()
    scipy_time = t1 - t0
    print(f"SciPy upfirdn time:  {scipy_time:.4f}s")
    
    # --- NumPy (no-JAX) ---
    _ = np.concatenate(list(upfirdn(h, [x[:100]], up=up, down=down, flush=True, np=np, jit=False))) # Warmup
    t0 = time.perf_counter()
    np_out = np.concatenate(list(upfirdn(h, [x], up=up, down=down, flush=True, np=np, jit=False)))
    t1 = time.perf_counter()
    np_time = t1 - t0
    print(f"chunkiter.upfirdn (no-JAX) time: {np_time:.4f}s")
    
    if has_jax:
        x_jax = jnp.asarray(x)
        
        # --- JAX Eager ---
        _ = jnp.concatenate(list(upfirdn(h, [x_jax[:100]], up=up, down=down, flush=True, np=jnp, jit=False))) # Warmup
        t0 = time.perf_counter()
        jax_eager_out = jnp.concatenate(list(upfirdn(h, [x_jax], up=up, down=down, flush=True, np=jnp, jit=False)))
        jax_eager_out.block_until_ready()
        t1 = time.perf_counter()
        jax_eager_time = t1 - t0
        print(f"JAX (Eager) time:    {jax_eager_time:.4f}s")
        
        # --- JAX JIT ---
        # Warmup for JIT (compile with EXACT SAME SHAPE to properly measure pure runtime)
        t0 = time.perf_counter()
        _ = jnp.concatenate(list(upfirdn(h, [x_jax], up=up, down=down, flush=True, np=jnp, jit=True)))
        _.block_until_ready()
        t1 = time.perf_counter()
        jax_jit_compilation_time = t1 - t0
        
        t0 = time.perf_counter()
        jax_jit_out = jnp.concatenate(list(upfirdn(h, [x_jax], up=up, down=down, flush=True, np=jnp, jit=True)))
        jax_jit_out.block_until_ready()
        t1 = time.perf_counter()
        jax_jit_time = t1 - t0
        print(f"JAX (JIT) time:      {jax_jit_time:.4f}s  (Compilation + first run: {jax_jit_compilation_time:.4f}s)")

if __name__ == "__main__":
    # Standard tests
    test_upfirdn_against_scipy()
    print("test_upfirdn_against_scipy passed")

    test_jax_upfirdn_against_scipy()
    print("test_jax_upfirdn_against_scipy passed")
    
    test_single_chunk_performance()
    print("test_single_chunk_performance completed")

    print("\nAll tests passed.")
