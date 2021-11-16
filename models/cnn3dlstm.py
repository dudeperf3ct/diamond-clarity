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
            self.cnn_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        if model_name == 'resnet18':
            model = resnet18(sample_size=sample_size, sample_duration=sample_duration)
            self.cnn_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        if model_name == 'simple_cnn3d':
            model = Simple3dCNN()
            self.cnn_model = torch.nn.Sequential(*(list(model.children())[:-4]))
        # self.lstm = nn.LSTM(input_size=self.cnn_model.fc.out_features, hidden_size=512, num_layers=3)
        # self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn_model(x)
        # x = self.lstm(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x


if __name__ == '__main__':
    d_in = torch.randn((1, 3, 6, 384, 384))
    m = CNN3DLSTM('resnet10', 384, 6)
    print(m)
    print(m(d_in).shape)