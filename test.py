import torch

mask = torch.randn(4).byte() % 2

print(mask)

crossentropy=torch.rand(4,1)

print(crossentropy.masked_select(mask).size())
