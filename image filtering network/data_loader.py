import torch
from torchvision import transforms, datasets

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing images to 256x256
    transforms.ToTensor(),          # Conversion to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize
])

# Load Dataset
def load_data(data_dir, batch_size):
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

