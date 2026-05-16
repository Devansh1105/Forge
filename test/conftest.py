import random

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(0)
    torch.manual_seed(0)
