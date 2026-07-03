"""
Hannes Robben 2025
MIRA - Microscopy Image Registration & Analysis


"""
from pathlib import Path

import numpy as np
from dataclasses import dataclass


@dataclass
class ImageData:
    def __init__(self, data: np.array | Path | str):
        self.data = data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size


class Structure(ImageData):
    def __init__(self,
                 data: np.array | Path | str,
                 ):
        super.__init__(data)


class Measurement(Structure):
    def __init__(self,
                 data: np.array | Path | str,
                 pixel_size: tuple | list | float,
                 metadata: dict,
                 intensity: np.array | Path | str = None,
                 *plot_args):
        super.__init__(data)


# wait for using this
class Processor:
    def __init__(self,
                 measurement_data: Measurement | list[Measurement] | tuple[Measurement],
                 information: dict,
                 ):
        pass
