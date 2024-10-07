import torch
from torchao.testing.utils import copy_tests, TorchAOTensorParallelTestCase
from torch.testing._internal.common_utils import run_tests
from torchao.quantization import int8_weight_only, float8_weight_only, float8_dynamic_activation_float8_weight
import unittest

class TestInt8woAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    QUANT_METHOD_FN = staticmethod(int8_weight_only)
copy_tests(TorchAOTensorParallelTestCase, TestInt8woAffineQuantizedTensorParallel, "int8wo_tp")

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)

class TestFloat8woAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    QUANT_METHOD_FN = staticmethod(float8_weight_only)
# @unittest.skipIf(is_H100, "Need H100")
copy_tests(TorchAOTensorParallelTestCase, TestFloat8woAffineQuantizedTensorParallel, "fp8wo_tp")

class TestFloat8dqAffineQuantizedTensorParallel(TorchAOTensorParallelTestCase):
    QUANT_METHOD_FN = staticmethod(float8_dynamic_activation_float8_weight)
# @unittest.skipIf(is_H100, "Need H100")
copy_tests(TorchAOTensorParallelTestCase, TestFloat8dqAffineQuantizedTensorParallel, "fp8dq_tp")

if __name__ == "__main__":
    run_tests()
