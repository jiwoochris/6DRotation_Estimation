from torchvision import models
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet50()  # we do not specify pretrained=True, i.e. do not load default weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)


# Define the same transform function as before
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img_path = 'e_data/5.png'
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # add batch dimension and send image to the device

# Make the prediction
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    output = model(image)
x_pred, y_pred = output[0]

x_pred = x_pred.item()
y_pred = y_pred.item()

print(x_pred, y_pred)






# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2

img = cv2.imread('e_data/5.png')

model = SixDRepNet()
model.draw_axis(img, 0, 25.558958574313053, 31.55232731345202)     # -25.558958574313053,-31.55232731345202  # 0, -y_pred, x_pred


cv2.imshow("test_window", img)
cv2.waitKey(0)
