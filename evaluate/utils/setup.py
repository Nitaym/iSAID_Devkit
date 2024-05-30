from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("cython_bbox", ["cython_bbox.c"],
                  include_dirs=[numpy.get_include()]),
        Extension("cython_nms", ["cython_nms.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# # Or, if you use cythonize() to make the ext_modules list,
# # include_dirs can be passed to setup()

# setup(
#     ext_modules=cythonize("cython_bbox.pyx"),
#     include_dirs=[numpy.get_include()]
# )    