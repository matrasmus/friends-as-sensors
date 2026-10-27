"""Multiprocessing helpers for the bootstrap analyses.

`parallel_process` maps a function over a list of items on several cores and
returns the results as a list, which the caller combines (pd.concat for frames,
merging for dicts). `data_division` splits a frame, dict or list into chunks.
"""

import pandas as pd
from multiprocess import Pool, cpu_count
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parallel map
# ---------------------------------------------------------------------------
def parallel_process(process_func, items, n_cores=None, show_progress=True,
                     ignore_errors=False, collect_results=True, **kwargs):
    """Apply `process_func` to every item in parallel and return the results.

    n_cores defaults to a quarter of the available cores. With ignore_errors,
    failed items are printed and dropped from the results. With
    collect_results=False, nothing is returned. Extra kwargs go to process_func.
    """
    if n_cores is None:
        n_cores = max(1, cpu_count() // 4)

    def wrapped_func(item):
        try:
            if kwargs:
                return process_func(item, **kwargs)
            return process_func(item)
        except Exception as e:
            if ignore_errors:
                print(f"Error processing item {item}: {str(e)}")
                return None
            raise e

    with Pool(processes=n_cores) as pool:
        if show_progress:
            if collect_results:
                results = list(tqdm(
                    pool.imap_unordered(wrapped_func, items),
                    total=len(items),
                    desc="Processing items"
                ))
            else:
                for _ in tqdm(
                    pool.imap_unordered(wrapped_func, items),
                    total=len(items),
                    desc="Processing items"
                ):
                    pass
                return None
        else:
            if collect_results:
                results = list(pool.imap_unordered(wrapped_func, items))
            else:
                for _ in pool.imap_unordered(wrapped_func, items):
                    pass
                return None

    if collect_results:
        if ignore_errors:
            results = [r for r in results if r is not None]
        return results
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def data_division(data, n_chunks, balanced=True):
    """Split a DataFrame, dict or list into n_chunks pieces of the same type.

    n_chunks is capped at the length of the data. With balanced, chunk sizes
    differ by at most one item.
    """
    def chunk_indices(length, n_chunks):
        if not balanced:
            chunk_size = length // n_chunks
            return [(i, i + chunk_size) for i in range(0, length, chunk_size)]

        base_size = length // n_chunks
        remainder = length % n_chunks

        indices = []
        current = 0
        for i in range(n_chunks):
            size = base_size + (1 if i < remainder else 0)  # spread the remainder
            indices.append((current, current + size))
            current += size

        return indices

    total_length = len(data)
    if n_chunks > total_length:
        n_chunks = total_length

    if isinstance(data, pd.DataFrame):
        return [
            data.iloc[start:end]
            for start, end in chunk_indices(len(data), n_chunks)
        ]

    elif isinstance(data, dict):
        items = list(data.items())
        return [
            dict(items[start:end])
            for start, end in chunk_indices(len(items), n_chunks)
        ]

    # any other sequence that supports len() and slicing
    return [
        data[start:end]
        for start, end in chunk_indices(len(data), n_chunks)
    ]
