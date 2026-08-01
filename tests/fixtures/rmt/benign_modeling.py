import os
import re

import pandas
import torch


RX = re.compile(r"[a-z]+")
TORCH_COMPILED = torch.compile


def helper(path=os.path.join("a", "b")):
    return path


def dataframe_eval(frame):
    return frame.eval("value + 1")


def pandas_eval():
    return pandas.eval("1 + 1")
