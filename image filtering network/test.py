import torch
import data_loader
from model_res import CNNModelWithResidual


data_dir = 'val'
batch_size = 32
val_loader = data_loader.load_data(data_dir, batch_size)


model = CNNModelWithResidual()

model.load_state_dict(torch.load('weights/best_weights.pt'))


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on validation set: {accuracy:.2%}')
