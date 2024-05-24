import torch

def add_noise(tensor, noise_level=0.01):

    value_range = tensor.max() - tensor.min()
    
    # Generate random noise scaled by the range of the tensor and the desired noise level
    noise = noise_level * value_range * torch.randn_like(tensor)
    
    # Create a mask for non-zero elements
    mask = tensor != 0
    
    # Apply noise only to non-zero elements
    tensor = tensor + noise * mask
    
    return tensor


# Example usage
tensor = torch.tensor([0.0, 20.0, 15.0, 25.0, 30.0])
noisy_tensor = add_noise(tensor, noise_level=0.05)  # 5% noise relative to the tensor's range

print(noisy_tensor)