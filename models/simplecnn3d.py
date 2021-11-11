# based on : https://keras.io/examples/vision/3D_image_classification/
import torch
import torch.nn as nn
import math

class Simple3dCNN(nn.Module):
    def __init__(self) -> None:
        super(Simple3dCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=11, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._make_conv_layer(64, 128, 7, (1, 2, 2), 3)
        self.conv3 = self._make_conv_layer(128, 256, 5, (1, 2, 2), 3)
        self.conv4 = self._make_conv_layer(256, 512, 5, (1, 2, 2), 3)
        # self.conv5 = self._make_conv_layer(512, 1024, 5, (1, 2, 2), 3)
        self.fc = nn.Linear(4608, 256)
    
    def _make_conv_layer(self, in_c, out_c, k, s, p):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.LeakyReLU(),
        nn.MaxPool3d(2),
        nn.BatchNorm3d(out_c)
        )
        return conv_layer

    def forward(self, x):
        # # print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        # x = self.conv5(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        # # print(x.size())
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.pool = nn.AdaptiveAvgPool3d(4)
        # self.conv_layer2 = self._conv_layer_set(64, 64)
        self.fc1 = nn.Linear(2048, 128)
        self.fc = nn.Linear(128, 100)
        self.relu = nn.LeakyReLU()
        # self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # print(x.size())
        out = self.conv_layer1(x)
        # print(out.size())
        out = self.pool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        # print(out.size())
        out = self.relu(out)
        # print(out.size())
        # out = self.batch(out)
        out = self.drop(out)
        # print(out.size())
        out = self.fc(out)
        # print(out.size())
        return out


# if __name__ == '__main__':
#     d_in = torch.randn(1, 3, 6, 384, 384)
#     # m = Simple3dCNN()
#     m = CNNModel()
#     print(m)
#     print(m(d_in).shape)