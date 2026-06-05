import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = FakeData(
    size=10000,
    image_size=(3, 256, 256),
    num_classes=10,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

device = "cuda"
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(3 * 224 * 224, 10)
).to(device)

start = time.time()

for i, (x, y) in enumerate(loader):
    if i >= 100:
        break
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    out = model(x)

torch.cuda.synchronize()
end = time.time()

print("Time:", end - start)
