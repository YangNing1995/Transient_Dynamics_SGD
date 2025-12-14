import torch
import torch.nn as nn
import torch.nn.init as init

# Two-layer Fully Connected Network
class FCN(nn.Module):
    def __init__(self, input_dim=768, hidden=50):
        super(FCN, self).__init__()
        self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden, bias=True), 
                nn.ReLU(),
                nn.Linear(hidden, hidden, bias=True),
                nn.ReLU(),
                nn.Linear(hidden, 10, bias=True), 
                )
        self.initialize_weights()
    def initialize_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self,x):
        output = self.net(x)
        return output

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=3072, hidden=512):
        """
        Args:
            input_dim: Flattened input dimension (e.g., 3072 for CIFAR10, 784 for MNIST)
            hidden: Number of neurons in the fully connected layer
        """
        super(SimpleCNN, self).__init__()
        
        if input_dim == 784:
            self.in_channels = 1
            self.img_size = 28
        elif input_dim == 3072:
            self.in_channels = 3
            self.img_size = 32
        else:
            import math
            self.img_size = int(math.sqrt(input_dim // 3)) 
            if self.img_size * self.img_size * 3 != input_dim:
                self.in_channels = 1
                self.img_size = int(math.sqrt(input_dim))
            else:
                self.in_channels = 3

        # Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool ...
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/2, W/2

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/4, W/4

            # Layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/8, W/8
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.img_size, self.img_size)
            dummy_output = self.features(dummy_input)
            self.flat_features = dummy_output.view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10) 
        )

        # Initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):            
        x = self.features(x)
        x = self.classifier(x)
        return x 