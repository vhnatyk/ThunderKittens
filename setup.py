import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from config import sources, target, kernels
target = target.lower()

# Set environment variables
thunderkittens_root = os.getenv('THUNDERKITTENS_ROOT', os.path.abspath(os.path.join(os.getcwd(), '.')))
python_include = subprocess.check_output(['python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
torch_include = subprocess.check_output(['python', '-c', "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"]).decode().strip()
torch_lib_dir = subprocess.check_output(['python', '-c', "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"]).decode().strip()
print('Thunderkittens root:', thunderkittens_root)
print('Python include:', python_include)
print('Torch include directories:', torch_include)
print('Torch lib directory:', torch_lib_dir)

# CUDA flags
cuda_flags = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-std=c++20',
    '-O3',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    f'-I{thunderkittens_root}/include',
    f'-I{thunderkittens_root}/prototype',
    f'-I{python_include}',
    '-DTORCH_COMPILE'
] + torch_include.split()
cpp_flags = [
    '-std=c++20',
    '-O3'
]

if target == '4090':
    cuda_flags.append('-DKITTENS_4090')
    cuda_flags.append('-arch=sm_89')
elif target == 'h100':
    cuda_flags.append('-DKITTENS_HOPPER')
    # IMPORTANT:
    # Do NOT pass '-arch=sm_90a' here.
    #
    # When building via torch.utils.cpp_extension, PyTorch will auto-inject
    # `-gencode` flags based on TORCH_CUDA_ARCH_LIST. We want that mechanism
    # so we can target Hopper "a" (sm_90a) with:
    #   TORCH_CUDA_ARCH_LIST=9.0a
    #
    # If we pass any flag containing 'arch', PyTorch disables its arch-flag
    # injection, and nvcc may emit PTX with `.target sm_90` which fails for
    # WGMMA instructions that require `.target sm_90a`.
elif target == 'a100':
    cuda_flags.append('-DKITTENS_A100')
    cuda_flags.append('-arch=sm_80')
else:
    raise ValueError(f'Target {target} not supported')

source_files = ['thunderkittens.cpp']
for k in kernels:
    if target not in sources[k]['source_files']:
        raise KeyError(f'Target {target} not found in source files for kernel {k}')
    if type(sources[k]['source_files'][target]) == list:
        source_files.extend(sources[k]['source_files'][target])
    else:
        source_files.append(sources[k]['source_files'][target])
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')

setup(
    name='thunderkittens',
    ext_modules=[
        CUDAExtension(
            'thunderkittens',
            sources=source_files, 
            extra_compile_args={'cxx' : cpp_flags,
                                'nvcc' : cuda_flags}, 
            libraries=['cuda'],
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)