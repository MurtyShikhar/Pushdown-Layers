import torch
import torch.utils.data
from .. import utils
import numpy as np
from typing import Dict, Any
import numpy as np



def get_randstate(seed: Optional[int] = None) -> np.random.RandomState:
    if seed is None:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.seed
        else:
            seed = random.randint(0, 0x7FFFFFFF)

    return np.random.RandomState(seed)


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=True, seed=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = get_randstate(seed)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            while True:
                yield self.seed.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = self.seed.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1
                yield sample

    def __len__(self):
        return 0x7FFFFFFF
