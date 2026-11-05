"""Process-local backport for legacy Numba on NVIDIA driver 580+.

Upstream fix:
https://github.com/NVIDIA/numba-cuda/commit/6e08c9d08e9de59c7af28b720289debbbd384764
The v2 context APIs have different signatures. Legacy Numba must use the
unversioned symbols matching its existing ctypes prototypes.
"""


def patch_legacy_context_api():
    import numba
    from numba.cuda.cudadrv.driver import Driver, USE_NV_BINDING

    version = tuple(int(part) for part in numba.__version__.split('.')[:2])
    if USE_NV_BINDING or version >= (0, 63):
        return
    original = Driver._find_api
    if getattr(original, '_find_context_api_backport', False):
        return

    def find_api(self, name):
        if name in ('cuCtxGetDevice', 'cuCtxSynchronize'):
            return getattr(self.lib, name)
        return original(self, name)

    find_api._find_context_api_backport = True
    Driver._find_api = find_api
