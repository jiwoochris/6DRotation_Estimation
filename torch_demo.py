from torchvision import models
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms



number = 12

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
img_path = f'e_data/{number}.png'
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # add batch dimension and send image to the device

# Make the prediction
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    output = model(image)
x_pred, y_pred = output[0]

x_pred = x_pred.item()
y_pred = y_pred.item()

print("\n\nprediction : ", x_pred, y_pred)



import pandas as pd

file = pd.read_csv('new_xy_data.csv')
print("label : ", file.iloc[number-1]['x'], file.iloc[number-1]['y'])


if x_pred > 0: x_pred_rep = 'left'
else : x_pred_rep = 'right'

if y_pred > 0: y_pred_rep = 'downside'
else : y_pred_rep = 'upside'

print(f'\n{abs(x_pred) : .3f} to the {x_pred_rep}, {abs(y_pred) : .3f} to the {y_pred_rep}')



import cv2

img = cv2.imread(f'e_data/{number}.png')

cv2.imshow("The larger X the more left, The larger Y more down", img)
cv2.waitKey(0)
