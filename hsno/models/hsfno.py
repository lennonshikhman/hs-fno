from __future__ import annotations
import torch.nn as nn
from .hsno import HSNO


class HistorySpaceFNO(HSNO):
    """HS-FNO: History-Space Fourier Neural Operator.

    The model consumes a full history state, concatenates conditioning variables
    as constant channels, parameterizes the exposed-future predictor with an
    FNO-style spectral backbone, and updates the history using exact
    shift-append transport.
    """

    def __init__(self, *args, backbone: str = "fno", **kwargs):
        super().__init__(*args, backbone="fno", **kwargs)


# Backward-friendly alias for paper-facing prose.
HSFNO = HistorySpaceFNO
