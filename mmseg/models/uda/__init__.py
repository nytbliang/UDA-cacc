# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from .gan import GAN
from .multi_ca import MultiCA
from .multi_da import MultiDA
from .baseline import BASELINE

__all__ = ['DACS', 'GAN', 'MultiDA', 'BASELINE', 'MultiCA']
