import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from model_res import CNNModelWithResidual

# Load trained model weights
model_path = 'weights/best_weights.pt'
model = CNNModelWithResidual()
model.load_state_dict(torch.load(model_path))
model.eval()

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Assuming the path where the categorized images are to be saved
output_dir_exposure = 'detect/exposure/'  # Save path with flare interference image
output_dir_non = 'detect/non/'  # Save path with no flare interference image

os.makedirs(output_dir_exposure, exist_ok=True)
os.makedirs(output_dir_non, exist_ok=True)

# Load the folder of images to be predicted
input_images_dir = ''  # Replace with the folder path of the image to be detected


# Get a list of image files to be predicted (including multi-layer folder structure)
image_paths = []
for root, dirs, files in os.walk(input_images_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_paths.append(os.path.join(root, file))

# Iterate over each image for prediction and classification
for image_path in image_paths:
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Predictions using models
    with torch.no_grad():
        output = model(input_tensor)

    # Access to forecasts
    probabilities = torch.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    # Move images to different folders based on predicted results
    if predicted_class == 0:
        shutil.copy(image_path, output_dir_exposure)
    else:
        shutil.copy(image_path, output_dir_non)
