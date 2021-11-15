import torch
import torch.nn as nn

from convlstmlayer import ConvLSTM2D


class ConvLSTM2DModel(nn.Module):
    def __init__(self) -> None:
        super(ConvLSTM2DModel, self).__init__()
        # input size for convlstm2d: (B, T, C, H, W) or (T, B, C, H, W)
        self.layer1 = ConvLSTM2D(3, 16, 3, 1, True, True, False)

    def forward(self, x):
        x = self.layer1(x)
        return x


# if __name__ == '__main__':
#     d_in = torch.randn((1, 6, 3, 384, 384))
#     m = ConvLSTM2DModel()
#     print(m)
#     print(m(d_in).shape)