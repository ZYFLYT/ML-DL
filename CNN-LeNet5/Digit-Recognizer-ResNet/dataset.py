import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class LeafDataset(Dataset):
    