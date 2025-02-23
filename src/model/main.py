import io
import os
from collections import OrderedDict
from typing import Any

import torch
from ruamel.yaml.compat import ordereddict
from torch import Tensor
from torchvision import models

import exportsd
import importsd

script_path = os.path.abspath(__file__)
cache_path = os.path.dirname(script_path)
torch.hub.set_dir(cache_path)
dats_path = os.path.join(cache_path, 'dats')

cpu_device = torch.device('cpu')
is_support_cuda = False
cuda_device = torch.device('cpu')

# Check for device availability
if torch.cuda.is_available():
    print("The current device supports GPU.")
    is_support_cuda = True
    cuda_device = torch.cuda.current_device()
    print(f"The GPU in use is: {cuda_device}")
# elif torch.backends.mps.is_available():
#     print("The current device supports MPS.")
#     is_support_guda = True
#     cuda_device = torch.device('mps')
else:
    print("No GPU support, using CPU.")
    is_support_cuda = False
    current_device = torch.device('cpu')

torch.set_default_device(cuda_device)


def import_dat(dat_file_path: str):
    stream = io.open(dat_file_path, mode='r', encoding='utf-8')
    dict = importsd.load_state_dict(stream)
    return dict

def export_dat(model:  torch.nn.Module, model_name: str):
    cpu_filepath = os.path.join(dats_path, f"{model_name}.dat")
    if os.path.exists(cpu_filepath):
        os.remove(cpu_filepath)
    with open(cpu_filepath, "wb") as f_cpu:
        exportsd.save_state_dict(model.to(cpu_device).state_dict(), f_cpu)


def export_onnx(model: OrderedDict[Any, Tensor], onnx_file_path: str, shape):
    # Create a sample input tensor (consistent with the expected input shape of the model)
    # 创建一个示例输入张量（与模型预期的输入形状一致）
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出模型
    torch.onnx.export(
        model,  # 要转换的 PyTorch 模型
        dummy_input,  # 示例输入数据
        onnx_file_path,  # ONNX 文件保存路径
        export_params=True,  # 是否导出训练参数
        opset_version=11,  # ONNX 操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 可选：用于动态批大小
    )

    print(f"模型成功导出到 {onnx_file_path}")


print("All models have been exported.")
