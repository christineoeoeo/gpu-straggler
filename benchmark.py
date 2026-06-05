import torch
import time

device = "cuda"

x = torch.randn(4096, 4096, device=device)

torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    y = torch.matmul(x, x)

torch.cuda.synchronize()
end = time.time()

print("Time:", end - start)
