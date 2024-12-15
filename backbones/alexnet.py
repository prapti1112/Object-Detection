"""Backbone alxenet model for image classification or feature extraction. 
This is an implementation from https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096*6*6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        logger.debug(f"Feature extraction: {x.shape}")
        x = torch.flatten(x, 1)
        logger.debug(f"Falttened output: {x.shape}")
        x = self.classifier(x)
        x = nn.Linear(4096, self.num_classes)(x)
        logger.debug(f"Classifier output: {x.shape}")

        return x
    
    def get_features(self, x):
        x = self.feature_extractor(x)
        logger.debug(f"Feature extraction: {x.shape}")
        x = torch.flatten(x, 1)
        logger.debug(f"Flattened output: {x.shape}")
        x = self.classifier(x)
        logger.debug(f"Classifier output: {x.shape}")
        return x

# Training example
def train():
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    transform = transforms.Compose([ 
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
   

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = AlexNet(len(classes))
    logger.debug(f"Model Parameters: {model.parameters()}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2): 
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100:
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    logger.info('Finished Training')



if __name__ == "__main__":
    train()