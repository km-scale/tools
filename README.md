# tools

## Zarr utils

The `zarrutils.py` module contains helper functions for creating Zarr stores.
Notably, it provides the convenience function `get_encoding()`,
which takes an Xarray dataset as input and provides a complete Zarr v3 encoding.

```python
from zarrutils import get_encoding

ds.to_zarr("test.zarr", encoding=get_encoding(ds), zarr_format3)
```
