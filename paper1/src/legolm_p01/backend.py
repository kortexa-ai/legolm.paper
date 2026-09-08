"""Numerical backend identity, independent of dataset and model identity."""
import platform

import torch


def identity(device):
    device=torch.device(device)
    result={"device_type":device.type,"torch_version":torch.__version__,"cpu_machine":platform.machine(),
            "intraop_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "execution":"FP32, autocast disabled, TF32 disabled"}
    if device.type=="cuda":
        props=torch.cuda.get_device_properties(device)
        result.update({"device_name":props.name,"compute_capability":[props.major,props.minor],
                       "device_total_bytes":props.total_memory,"cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version()})
    elif device.type=="mps":
        result["macos_version"]=platform.mac_ver()[0]
    return result
