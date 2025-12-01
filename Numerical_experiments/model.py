import torch.nn as nn
import torch.nn.init as init

# Two-layer Fully Connected Network
class FCN(nn.Module):
    def __init__(self, hidden = 50):
        super(FCN,self).__init__()
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, hidden, bias = True),
                nn.ReLU(),
                nn.Linear(hidden, hidden, bias = True),
                nn.ReLU(),
                nn.Linear(hidden, 10, bias = True),
                )
        self.initialize_weights()

    def initialize_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Gaussian initialization with given std
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         init.normal_(m.weight, mean=0, std=0.2)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0.0)

    def forward(self,x):
        output = self.net(x)
        return output