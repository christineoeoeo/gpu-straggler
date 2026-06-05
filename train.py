import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

device = "cuda"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

trainset = datasets.CIFAR10(
    root="/tmp/kgzh0394/cifar10",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

model = resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("GPU:", torch.cuda.get_device_name(0))

for epoch in range(1):

    epoch_start = time.time()

    for i, (images, labels) in enumerate(loader):

        iter_start = time.time()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()

        iter_end = time.time()

        if i % 100 == 0:
            print(
                f"Batch {i}, "
                f"Time={iter_end - iter_start:.4f}s, "
                f"Loss={loss.item():.4f}"
            )

    print(
        "Epoch time:",
        time.time() - epoch_start
    )
