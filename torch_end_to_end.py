import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        labels = self.df.iloc[idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float)
        return image, labels

# Load your labels
df = pd.read_csv('new_xy_data.csv')

# Preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split the data into training and validation sets
train_df = df.sample(frac=0.8, random_state=0)
val_df = df.drop(train_df.index)

# Create the datasets
train_dataset = ImageDataset(train_df, 'NewDatasetProduced0', transform)
val_dataset = ImageDataset(val_df, 'NewDatasetProduced0', transform)





from torchvision import models
import torch.nn as nn

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # replace the last fc layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device : ", device)

model = model.to(device)




import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)





from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import copy

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')

no_improve = 0
early_stop = 5
num_epochs = 20

for epoch in range(num_epochs):
    
    model.train()

    train_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    print(f'Epoch {epoch}/{num_epochs-1}, Train Loss: {train_loss}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch}/{num_epochs-1}, Validation Loss: {val_loss}')

    # Save the model if it has the best validation loss so far
    if val_loss < best_loss:
        print(f'Saving model at epoch {epoch}')
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        # Early stopping
        if no_improve >= early_stop:
            print("Validation loss has not improved for 5 epochs, stopping.")
            model.load_state_dict(best_model_wts)
            break

# Load the best weights into the model
model.load_state_dict(best_model_wts)

torch.save(best_model_wts, 'best_model.pth')









