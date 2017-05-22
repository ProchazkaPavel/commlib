from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = [
      Extension("_Turbo_lib", sources=["_Turbo_lib.pyx", "Turbo_lib.c"],
                 include_dirs=[numpy.get_include()],
		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-Ofast", "-msse", "-msse2"]),                 
      Extension("_Sparse_Turbo_lib", sources=["_Sparse_Turbo_lib.pyx", "Sparse_Turbo_lib.c"],
                 include_dirs=[numpy.get_include()],
		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-Ofast", "-msse",  "-msse2", "-I."]),
                 ],
)

#		 extra_compile_args=["-lm", "-fno-stack-protector", "-Wcpp", "-O3", "-msse2", "-S"]),
#		 extra_compile_args=["-Ofast", "-lm", "-fno-stack-protector", "-Wcpp"]),
