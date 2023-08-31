from typing import List, Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple
from torch.nn import functional as F
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                            _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)
from torch import float32

class _RankPoolNd(Module):
    __constants__ = ['rank', 'kernel_size', 'stride', 'padding', 'return_indices']
    
    def __init__(self, rank=1, kernel_size=3, stride=1, padding=1, return_indices=True):
        super(_RankPoolNd, self).__init__()
        self.rank=rank,
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding=padding
        self.return_indices=return_indices


    def extra_repr(self) -> str:
        return 'rank={}, kernel_size={}, stride={}, padding={}, return_indices={}'.format(
            self.rank, self.kernel_size, self.stride, self.padding, self.return_indices
        )

class RankPool1d(_RankPoolNd):
    rank: _size_1_t
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    mode: str
    value: float32

    def __init__(self, rank=1, kernel_size=3, stride=1, padding=1, mode:str='same', value:float32=None, return_indices=True):
        assert kernel_size % 2 == 0, f'kernel_size must be odd, and {kernel_size} is not odd.'
        super(RankPool1d, self).__init__(rank=rank,
                                         kernel_size=kernel_size, 
                                         stride=stride, 
                                         padding=padding, 
                                         return_indices=return_indices
        )
        self.mode = mode
        if mode=='constant' and value==None:
            raise ValueError('Received padding mode \`constant\' but did not receive a value')
        self.value=value

    def forward(self, input: Tensor):
        ppair = _pair(self.padding)

        if self.value == None:
            p = F.pad(input, pad=ppair, mode=self.mode)
        else:
            p = F.pad(input, pad=ppair, mode=self.mode, value=self.value) 

        s = p.unfold(0, size=self.kernel_size, step=self.stride).kthvalue(k=self.rank[0], dim=1)
        return s if self.return_indices else s.values