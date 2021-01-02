from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "_Turbo_lib",
        ["_Turbo_lib.pyx",  "Turbo_lib.c"],
        include_dirs=[numpy.get_include()],
        #include_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
        #libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 'fftw3f_threads', 'fftw3l_threads'],
        #library_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
    ),
#    Extension(
#        "_Sparse_Turbo_lib",
#        ["_Sparse_Turbo_lib.pyx", "Sparse_Turbo_lib.c"],
#        #library_dirs=['.'], # not needed for fftw unless it is installed in an unusual place
#        libraries=['Sparse_Turbo_lib'],
#        #include_dirs=[''], # not needed for fftw unless it is installed in an unusual place
#        language="c"
#        )
]

setup(
    name = "TurboLib",
    version = "0.5.0",
    packages = find_packages(),
    ext_modules = cythonize(extensions),
    author="pavel prochazka",
    author_email="pavel@prochazka.info",
    description="Tools for creating and analysis of turbo codes"
)


#from distutils.core import setup, Extension
#import numpy
#from Cython.Distutils import build_ext

#      Extension("_Turbo_lib", sources=["_Turbo_lib.pyx", "Turbo_lib.c"],
#                 include_dirs=[numpy.get_include()],
##		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-Ofast", "-msse", "-msse2"]),                 
#      Extension("_Sparse_Turbo_lib", sources=["_Sparse_Turbo_lib.pyx", "Sparse_Turbo_lib.c"],
#                 include_dirs=[numpy.get_include()],
#		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-Ofast", "-msse",  "-msse2", "-I."]),
#                 ],
#)
#
#		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-O3", "-msse2", "-S"]),
#		 extra_compile_args=["-Ofast", "-lm", "-fno-stack-protector", "-Wcpp"]),
