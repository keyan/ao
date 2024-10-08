# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch

from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    get_groupwise_affine_qparams,
)
from .api import (
    FakeQuantizeConfig,
    QuantizationGranularity,
)
from .utils import (
    _choose_qparams_per_token_asymmetric,
    _fake_quantize_per_channel_group,
    _fake_quantize_per_token,
    _get_qmin_qmax,
)


class FakeQuantizer(torch.nn.Module):
    """
    Generic module for applying fake quantization to a tensor, as specified in the config.
    """
    def __init__(self, config: FakeQuantizeConfig):
        super().__init__()
        self.config = config
        self.enabled = True
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

        # TODO: support range learinng
        if self.config.range_learning:
            raise NotImplementedError("Range learning is not supported yet")

    def forward(self, x: torch.Tensor):
        """
        Apply fake quantization to the tensor based on the bit-width,
        granularity, symmetry, and other properties specified in the config.
        """
        if not self.enabled:
            return x

        if self.config.granularity == QuantizationGranularity.PER_TOKEN:
            return self._per_token_forward(x)
        elif self.config.granularity in [
            QuantizationGranularity.PER_CHANNEL,
            QuantizationGranularity.PER_GROUP,
        ]:
            return self._per_channel_or_group_forward(x)
        else:
            raise ValueError("Unknown granularity %s" % self.config.granularity)

    def _per_token_forward(self, x: torch.Tensor):
        """
        Perform per token fake quantization on the tensor.
        """
        if self.config.symmetric:
            raise NotImplementedError("Symmetric per token is not supported yet")
        if self._should_compute_qparams():
            (self.scale, self.zero_point) = _choose_qparams_per_token_asymmetric(
                x, self.config.scale_precision, self.config.zero_point_precision,
            )
        qmin, qmax = _get_qmin_qmax(self.config.bit_width)
        return _fake_quantize_per_token(x, self.scale, self.zero_point, qmin, qmax)

    def _per_channel_or_group_forward(self, x: torch.Tensor):
        """
        Perform per channel or per group fake quantization on the tensor.
        We express per channel using per group where the group size is the size
        of the last dimension of the tensor.
        """
        bit_width = self.config.bit_width
        granularity = self.config.granularity
        scale_precision = self.config.scale_precision
        zero_point_precision = self.config.zero_point_precision
        zero_point_domain = self.config.zero_point_domain
        symmetric = self.config.symmetric

        # get group size
        if granularity == QuantizationGranularity.PER_CHANNEL:
            group_size =  x.size()[-1]
        elif granularity == QuantizationGranularity.PER_GROUP:
            assert self.config.group_size is not None
            group_size = self.config.group_size
        else:
            raise ValueError("Group size not defined for granularity %s" % granularity)

        # get scales and zero points
        if self._should_compute_qparams():
            if symmetric:
                (self.scale, self.zero_point) = get_group_qparams_symmetric(
                    x, bit_width, group_size, scale_precision,
                )
            else:
                (self.scale, self.zero_point) = get_groupwise_affine_qparams(
                    x, bit_width, group_size, scale_precision,
                )
            self.zero_point = self.zero_point.to(zero_point_precision)

        qmin, qmax = _get_qmin_qmax(bit_width, symmetric)
        return _fake_quantize_per_channel_group(
            x, self.scale, self.zero_point, qmin, qmax, group_size, zero_point_domain,
        )

    def _should_compute_qparams(self) -> bool:
        """
        Return whether we need to compute new scales and zero points.
        """
        return self.config.dynamic or self.scale is None or self.zero_point is None
