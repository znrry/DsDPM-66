import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import data_loader
from model_res import CNNModelWithResidual
import time

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Setting hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 50


# Load data
data_dir = 'data'  # Replace with your data path
train_loader = data_loader.load_data(data_dir, batch_size)
weights_folder = 'weights'
os.makedirs(weights_folder, exist_ok=True)

# Initialize the model and move it to the GPU
model = CNNModelWithResidual().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Defining the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Every 5 epochs, the learning rate decays to a factor of 0.1 of the original

best_model_weights = None
best_loss = float('inf')  # Initialize to a larger value
last_weight_path = os.path.join(weights_folder, 'last_weights.pt')
best_weight_path = os.path.join(weights_folder, 'best_weights.pt')

# training model
losses = []  # Used to store the loss values for each epoch
for epoch in range(epochs):
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    end_time = time.time()
    epoch_time = end_time - start_time
    average_loss = running_loss / len(train_loader)
    losses.append(average_loss)
    time_str = time.strftime('%M:%S', time.gmtime(epoch_time))
    print(f"Epoch [{epoch + 1}/{epochs}] finished. Time taken: {time_str}, Average Loss: {average_loss:.4f}")

    # Preservation of optimal weights
    if average_loss < best_loss:
        best_loss = average_loss
        best_weight = model.state_dict().copy()

# Save final and optimal weights
torch.save(model.state_dict(), last_weight_path)
if best_weight:
    torch.save(best_weight, best_weight_path)

print(f"Last weight saved at '{last_weight_path}'")
if best_weight:
    print(f"Best weight saved at '{best_weight_path}'")

# Call the function that plots the loss curve after training is complete
model.plot_training_loss(losses)


print('Finished Training')
