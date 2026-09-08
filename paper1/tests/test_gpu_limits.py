import os
import unittest
from unittest.mock import patch

from legolm_p01.gpu_limits import configure


class GpuLimits(unittest.TestCase):
    def test_cpu_execution_does_not_inspect_or_initialize_cuda(self):
        with patch("torch.cuda.device_count",side_effect=AssertionError("CUDA touched")):
            configure("cpu")

    def test_unpinned_multiple_gpus_are_rejected(self):
        with patch("importlib.util.find_spec",return_value=None),patch("torch.cuda.device_count",return_value=2):
            with self.assertRaisesRegex(RuntimeError,"exactly one"):
                configure()

    def test_clean_public_checkout_applies_declared_caps(self):
        with patch.dict(os.environ,{"PAPER1_GPU_ALLOCATOR_FRACTION":"0.25","PAPER1_GPU_PRELOAD_MIN_FREE_GIB":"40"},clear=True),\
             patch("importlib.util.find_spec",return_value=None),patch("torch.cuda.device_count",return_value=1),\
             patch("torch.cuda.set_per_process_memory_fraction") as cap,patch("torch.cuda.mem_get_info",return_value=(50*1024**3,96*1024**3)):
            configure()
            cap.assert_called_once_with(.25,0)

    def test_insufficient_memory_aborts_before_model_loading(self):
        with patch.dict(os.environ,{},clear=True),patch("importlib.util.find_spec",return_value=None),\
             patch("torch.cuda.device_count",return_value=1),patch("torch.cuda.set_per_process_memory_fraction"),\
             patch("torch.cuda.mem_get_info",return_value=(9*1024**3,96*1024**3)):
            with self.assertRaises(MemoryError):
                configure()


if __name__=="__main__":
    unittest.main()
