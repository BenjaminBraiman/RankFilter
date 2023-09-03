from typing import List, Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple
from torch.nn import functional as F
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                            _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)
from torch import float32

class _RankFilterNd(Module):
    __constants__ = ['rank', 'kernel_size', 'stride', 'padding', 'return_indices']
    
    def __init__(self, rank=1, kernel_size=3, output=None, return_indices=True):
        super(_RankFilterNd, self).__init__()
        self.rank=rank,
        self.kernel_size = kernel_size
        self.stride=1
        self.return_indices=return_indices
        self.output = output
        if output != None:
            assert type(self.output) == Tensor
        

    def extra_repr(self) -> str:
        return 'rank={}, kernel_size={}, stride={}, padding={}, return_indices={}'.format(
            self.rank, self.kernel_size, self.stride, self.padding, self.return_indices
        )

class RankFilter1d(_RankFilterNd):
    rank: int
    kernel_size: int
    stride: int = 1
    mode: str
    value: float32
    padding: float32

    def __init__(self, rank=1, kernel_size=3, mode:str='constant', value:float32=0.0, return_indices=True):
        assert kernel_size % 2 == 1, f'kernel_size must be odd, and {kernel_size} is not odd.'
        super(RankFilter1d, self).__init__(rank=rank,
                                         kernel_size=kernel_size, 
                                         return_indices=return_indices,
        )
        self.mode = mode
        if mode=='constant' and value==None:
            raise ValueError('Received padding mode \'constant\' but did not receive a value')
        
        self.padding = self.kernel_size//2
        self.value=value

    def forward(self, input: Tensor):
        ppair = _pair(self.padding)

        if not(self.mode == 'constant'):
            p = F.pad(input, pad=ppair, mode=self.mode)
        else:
            p = F.pad(input, pad=ppair, mode=self.mode, value=self.value) 

        s = p.unfold(0, size=self.kernel_size, step=self.stride).kthvalue(k=self.rank[0], dim=1)
        if self.output != None:
            self.output[:] = s.values
        else:
            return s if self.return_indices else s.values
    
class RankFilter2d(_RankFilterNd):
    rank: int
    kernel_size: _size_2_t
    stride: int = 1
    mode: str
    value: float32
    padding_mode: str

    def __init__(self, rank=1, kernel_size=3, mode:str='constant', value:float32=0.0, output:Tensor=None, return_indices=True):
        
        if type(kernel_size) is int:
            kernel_size = _pair(kernel_size)
        elif len(kernel_size) != 2:
            raise ValueError('Kernel size must be a 2d integer tuple')

        for k in kernel_size:
            assert k % 2 == 1, f'kernel_size must be odd, and {k} is not odd.'
        
        self.mode = mode
        super(RankFilter2d, self).__init__(rank=rank,
                                         kernel_size=kernel_size, 
                                         return_indices=return_indices,
                                         output=output
                                         )
        self.value = value 

    def forward(self, input: Tensor):
        d0 = self.kernel_size[0]//2
        d1 = self.kernel_size[1]//2

        ppair = (d0, d0, d1, d1)

        if not(self.mode == 'constant'):
            p = F.pad(input, pad=ppair, mode=self.mode)
        else:
            p = F.pad(input, pad=ppair, mode=self.mode, value=self.value) 

        s = F.unfold(p, kernel_size=self.kernel_size, stride=self.stride).kthvalue(k=self.rank[0], dim=0)
        s.values.reshape(input.shape)
        if self.output != None:
            self.output[:] = s.values.reshape(input.shape)
        else:
            return s if self.return_indices else s.values 
        #return s if self.return_indices else s.values.reshape(input.shape)
## END    
