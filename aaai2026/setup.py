
from distutils.core import setup, Extension
module = Extension(
    "atcoder",
    sources=["atcoder_library_wrapper.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++14"]
)
setup(
    name="atcoder-library",
    version="0.0.1",
    description="wrapper for atcoder library",
    ext_modules=[module]
)
