import torch
import torch.nn as nn

from convlstmlayer import ConvLSTM2D


class ConvLSTM2DModel(nn.Module):
    def __init__(self) -> None:
        super(ConvLSTM2DModel, self).__init__()
        # input size for convlstm2d: (B, T, C, H, W) or (T, B, C, H, W)
        self.layer1 = ConvLSTM2D(input_dim, hidden_dim, kernel_size, num_layers)