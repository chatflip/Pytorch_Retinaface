from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Bbox:
    top_right: npt.NDArray[np.uint64]
    bottom_left: npt.NDArray[np.uint64]


@dataclass
class Face:
    bbox: Bbox
    confidence: float
    left_eye: npt.NDArray[np.uint64]
    right_eye: npt.NDArray[np.uint64]
    nose: npt.NDArray[np.uint64]
    mouth_right: npt.NDArray[np.uint64]
    mouth_left: npt.NDArray[np.uint64]
