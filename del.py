import torch

def set_nan_with_probability(tensor, probability):
    mask = torch.rand(tensor.shape) < probability
    tensor[mask] = float('nan')
    return tensor

# Example usage
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
probability = 0.3

result = set_nan_with_probability(tensor, probability)
print(result)