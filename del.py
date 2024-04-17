
import torch
import pdb

dim = 2

# Example tensor X of size [64, 6]
X = torch.Tensor([[1,2,3,4,5,6], [10,20,30,40,50,60]])

# Specified indices where X's data will be placed in the new tensor
indices = [0, 4, 2, 7, 3, 8]
# Initialize a new tensor of zeros with shape [64, 12]
result = torch.zeros(dim, 12)

# Place the data from X into the specified indices of the new tensor
for i, idx in enumerate(indices):
    result[:, idx] = X[:, i]




# Specified indices where X's data will be placed in the new tensor
indices = torch.tensor([0, 4, 2, 7, 3, 8])

# Initialize a new tensor of zeros with shape [64, 12]
result2 = torch.zeros(dim, 12)

# Use advanced indexing to place the data from X into the specified indices of the new tensor
result2[:, indices] = X




pdb.set_trace()