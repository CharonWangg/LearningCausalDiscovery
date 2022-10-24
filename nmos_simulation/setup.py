from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
# import os
# os.environ['CFLAGS'] = '-stdlib=libc++'

setup(name='sim2600',
      version='0.2',
      description='Simulation of Atari 2600',
      author='Greg Jones',
      packages=['sim2600'],
      include_dirs=[np.get_include()],
      ext_modules = cythonize(Extension("sim_ext",["sim2600/mycircuitsimulator.pyx",
                               "sim2600/cirsim.cc"],
                              language='c++',
                              extra_compile_args=['-O2', '-g'],)
                              ),
      package_data={'sim2600': ['chips/*', 'roms/*']},
      install_requires=open('requirements.txt').read().splitlines(),
      )


