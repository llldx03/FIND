"""FIND package and the CUDA operators used by SECOND-IoU."""
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def extension(module, sources):
    return CUDAExtension(
        name=f'pcdet.ops.{module}.{module}_cuda',
        sources=[str(Path('pcdet') / 'ops' / module / 'src' / name) for name in sources],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
    )


setup(
    name='find3d', version='1.0.0',
    description='Frequency Invariance Disentanglement for Test-Time Adaptation in LiDAR 3D Detection',
    author='Dongxiao Li, Zixuan Hu, Ling-Yu Duan',
    license='MIT AND Apache-2.0', python_requires='>=3.8',
    license_files=['LICENSE', 'NOTICE', 'LICENSES/Apache-2.0.txt'],
    packages=find_packages(),
    package_data={
        'pcdet.datasets.kitti.kitti_object_eval_python': ['LICENSE'],
        'tools.train_utils.optimization': ['LICENSE'],
    },
    ext_modules=[
        extension('iou3d_nms', ['iou3d_cpu.cpp', 'iou3d_nms_api.cpp', 'iou3d_nms.cpp', 'iou3d_nms_kernel.cu']),
        extension('roiaware_pool3d', ['roiaware_pool3d.cpp', 'roiaware_pool3d_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension},
)
