import torch
from torch import nn
from collections import OrderedDict

class svhn_cnn(nn.Module):

    """
    Classical CNN model for digit detection. A bit more careful on the arg and return types
    """
    def __init__(self, in_channels : int, out_dim : int) -> None:

        super().__init__()

        self._cnnlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,16,3, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )

        self._cnnlayer2 = nn.Sequential(
            nn.Conv2d(32,64,3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )

        self._cnnlayer3 = nn.Sequential(
            nn.Conv2d(128,256,3, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2)
            )

        # Shape of the final layer here is (256,4,4), therefore
        # for the linear layers we account for this

        self._dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*256,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,out_dim)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self._cnnlayer1(x)
        x = self._cnnlayer2(x)
        x = self._cnnlayer3(x)
        x = self._dense(x)

        # Now map the output of linear via softmax to a given class
        logits = nn.functional.softmax(x, dim=1)

        return logits
