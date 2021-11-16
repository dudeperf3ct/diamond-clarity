import torch
import torch.nn as nn

from resnet import resnet10, resnet18
from simplecnn3d import Simple3dCNN

class CNN3DLSTM(nn.Module):
    def __init__(self, model_name, sample_size, sample_duration) -> None:
        super(CNN3DLSTM, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet10':
            model = resnet10(sample_size=sample_size, sample_duration=sample_duration)
        if model_name == 'resnet18':
            model = resnet18(sample_size=sample_size, sample_duration=sample_duration)
        if model_name == 'simplecnn3d':
            model = Simple3dCNN()
        # remove last fc layer from all models
        self.cnn_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # out = (1, 512, 1, 1)
        x = self.cnn_model(x)
        # input to lstm (n, seq_len, ftrs) in lstm input_size=ftrs
        x = x.view(x.size(0), -1, x.size(1))
        x, _ = self.lstm(x)
        x = self.fc1(x[-1, :, :])
        x = self.fc2(x)
        return x


# if __name__ == '__main__':
#     d_in = torch.randn((1, 3, 6, 384, 384))
#     m = CNN3DLSTM('simple_cnn3d', 384, 6)
#     print(m)
#     print(m(d_in).shape)