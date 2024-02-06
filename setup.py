import os
from setuptools import setup
from torch.utils import cpp_extension

from glob import glob

sources = glob('src/*.cpp')

setup(
    name="extension",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="extension",
            sources=sources,
            include_dirs=[
                os.path.abspath("include"),
            ],
            extra_compile_args={
                "cxx": [
                    # "-Wall",
                    # "-Wextra",
                    "-std=c++17",
                    "-O3",
                    "-m64",
                    "-fvisibility=hidden"
                ],
                "nvcc": [
                    # "-Wno-deprecated-gpu-targets",
                    "-O3",
                    # "-Xcompiler",
                    # "-Wall",
                    "-std=c++17"
                ]
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
