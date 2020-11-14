import torch

a = torch.randn(2, 4)
b = torch.randn(2, 4)

print(torch.gt(a, b))
idxs = torch.nonzero(torch.gt(a, b), as_tuple=False)
print(idxs)


