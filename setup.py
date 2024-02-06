import os
from setuptools import setup
from torch.utils import cpp_extension

from glob import glob

def get_files(directory):
    cpp_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                cpp_files.append(os.path.join(root, file))
        for dir in dirs:
            cpp_files.extend(get_files(os.path.join(root, dir)))
    return cpp_files
            

sources = list(set(get_files('src')))
print(sources)

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
