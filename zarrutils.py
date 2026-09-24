import healpix as hp
import numpy as np
import zarr


def _closest_power_of_two(x):
    """Return closest power of 2."""
    return int(2 ** np.round(np.log2(x)))


def get_chunks(sizes, chunksize=2**22):
    """Return multi-dimensional chunks of roughly 16MB size (at 32bit precision).

    The goals of the chunking strategy are:
        - multi-dimensional chunks
        - aim for an uncompressed chunk size of around 16 MB
        - storing full global fields when fewer than 12 spatial chunks would be created
        - always choose time chunking as a power of two (2^n),
          to allow alignment between 2D and 3D arrays
    """
    match tuple(sizes.keys()):
        case ("time", "level", "cell"):
            cell_chunksize = int(4**7) if sizes["cell"] / 4**7 >= 12 else sizes["cell"]
            level_chunksize = 5

            chunks = {
                "time": _closest_power_of_two(
                    chunksize // (level_chunksize * cell_chunksize)
                ),
                "cell": cell_chunksize,
                "level": level_chunksize,
            }
        case ("time", "cell"):
            cell_chunksize = int(4**8) if sizes["cell"] / 4**8 >= 12 else sizes["cell"]

            chunks = {
                "time": _closest_power_of_two(chunksize // cell_chunksize),
                "cell": cell_chunksize,
            }
        case (single_dim,):
            chunks = {
                single_dim: sizes[single_dim],
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_shards(sizes):
    """Suggest a tuple of shard sizes for a given dict of array dimensions.

    The current implementation simply combines all `cell` and `level`
    into a single shard.
    """
    # Map dimension names to chunk sizes
    dim_to_chunk = dict(zip(sizes.keys(), get_chunks(sizes)))

    match tuple(sizes.keys()):
        case ("time", "level", "cell"):
            chunks = {
                "time": dim_to_chunk["time"],
                "level": sizes["level"],
                "cell": sizes["cell"],
            }
        case ("time", "cell"):
            chunks = {
                "time": dim_to_chunk["time"],
                "cell": sizes["cell"],
            }
        case (single_dim,):
            chunks = {single_dim: sizes[single_dim]}
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_compressor():
    """Return a numcodecs compressor."""
    return zarr.codecs.BloscCodec(cname="zstd", clevel=6)


def get_dtype(da):
    """Suggest a dtype for storage (e.g., float32 for floats)."""
    if np.issubdtype(da.dtype, np.floating):
        # Store all floating points dtype in single precision.
        return {"dtype": "float32"}
    else:
        # For other data types, do **not set** the `dtype` field at all.
        # This preserves the default encoding of, e.g., time values.
        return {}


def get_encoding(dataset):
    return {
        var: {
            "compressors": [get_compressor()],
            "chunks": get_chunks(dataset[var].sizes),
            "shards": get_shards(dataset[var].sizes),
            **get_dtype(dataset[var]),
        }
        for var in dataset.variables
    }
