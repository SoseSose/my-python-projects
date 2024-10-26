# %%
import random
import numpy as np
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def fix_random_seed(seed=42):
    """
    乱数生成器のシードを固定し、実験の再現性を確保します。

    この関数は以下のライブラリの乱数生成器を設定します：
    - Python の random モジュール
    - NumPy
    - PyTorch (CPU および CUDA)

    また、PyTorch の決定論的アルゴリズムを有効にします。

    パラメータ:
    - seed (int): 使用する乱数シード。デフォルトは42。

    注意:
    - この関数はグローバルな影響を持ちます。
    - PyTorch の cudnn の決定論的モードを有効にするため、パフォーマンスに影響を与える可能性があります。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    print("random seed fixed to {}. Warn !! It is Global".format(seed))


def set_torch_cuda_matmul_precision_to_high():
    """
    PyTorchのCUDA行列乗算の精度を設定します。

    この関数は、Tensor Coresを持つCUDAデバイスでのパフォーマンスを最適化するために、
    float32の行列乗算精度を'high'に設定します。

    注意:
    - この設定は精度とパフォーマンスのトレードオフに影響を与えます。
    - 'high'設定は、精度を犠牲にしてパフォーマンスを向上させる可能性があります。
    - この関数は、特定のCUDAデバイスに関する下記のWarningを回避するために使用されます。

    You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    """
    torch.set_float32_matmul_precision("high")


def ini_setting():
    fix_random_seed()
    set_torch_cuda_matmul_precision_to_high()


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
