import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 9 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 256 * 9 * 9)
        x = self.classifier(x)
        return x
