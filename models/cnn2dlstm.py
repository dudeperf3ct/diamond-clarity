import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CNN2DLSTM(nn.Module):
    def __init__(self, model_name):
        super(CNN2DLSTM, self).__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        if model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 256))
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, :, t, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

# if __name__ == '__main__':
#     d_in = torch.randn(1, 3, 6, 384, 384)
#     m = CNN2DLSTM('resnet18')
#     # print(m)
#     # print(m(d_in).shape)
#     m = CNN2DLSTM('resnet50')
#     # print(m)
#     print(m(d_in).shape)