# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchao.sparsity.prototype.training.autograd import semi_sparse_sparsify
from torchao.sparsity.prototype.training.pointwise_ops import CUTLASS_POINTWISE_OP_DISPATCH_TABLE

from torchao.quantization.utils import TORCH_VERSION_AFTER_2_2

# load pointwise op support, which exists only for CUTLASS
if TORCH_VERSION_AFTER_2_2:
    from torch.sparse import SparseSemiStructuredTensorCUTLASS
    SparseSemiStructuredTensorCUTLASS._load_dispatch_table(CUTLASS_POINTWISE_OP_DISPATCH_TABLE)

__all__ = [
    "SemiSparseLinear",
    "SemiSparseActivationLinear",
    "swap_linear_with_semi_sparse_linear_",
    "swap_semi_sparse_linear_with_linear_",
]

class SemiSparseLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime weight sparsity
    """

    def forward(self, x):
        sparse_weight = semi_sparse_sparsify(self.weight, backend="cusparselt")
        return torch.nn.functional.linear(x, sparse_weight, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @classmethod
    def to_dense(cls, semi_sparse_linear):
        mod = torch.nn.Linear(semi_sparse_linear.in_features, semi_sparse_linear.out_features)
        mod.weight = semi_sparse_linear.weight
        mod.bias = semi_sparse_linear.bias
        return mod


class SemiSparseActivationLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime activation sparsity
    """

    def forward(self, x):
        sparse_x = semi_sparse_sparsify(x, backend="cusparselt")
        return torch.nn.functional.linear(sparse_x, self.weight, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @classmethod
    def to_dense(cls, semi_sparse_linear):
        mod = torch.nn.Linear(semi_sparse_linear.in_features, semi_sparse_linear.out_features)
        mod.weight = semi_sparse_linear.weight
        mod.bias = semi_sparse_linear.bias
        return mod

def swap_linear_with_semi_sparse_linear_(model, config, current=""):
    """
    Public API for replacing nn.Linear with SemiSparseLinear
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, torch.nn.Linear):
            if fqn in config:
                setattr(model, name, config[fqn].from_dense(child))
                del child
        else:
            swap_linear_with_semi_sparse_linear_(child, config, current=fqn)

def swap_semi_sparse_linear_with_linear_(model, current=""):
    """
    Public API for replacing instances of SemiSparseLinear/SemiSparseActivaitonLinear with nn.Linear
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, (SemiSparseLinear, SemiSparseActivationLinear)):
            setattr(model, name, child.to_dense(child))
            del child
        else:
            swap_semi_sparse_linear_with_linear_(child, current=fqn)
