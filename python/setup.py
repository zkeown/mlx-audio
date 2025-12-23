"""Setup script for mlx-audio with C++ extension."""

from mlx import extension
from setuptools import setup

setup(
    ext_modules=[extension.CMakeExtension("mlx_audio.primitives._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
)
