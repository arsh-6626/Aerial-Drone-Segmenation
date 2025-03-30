import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import *
from dataloader import *
torch.cuda.empty_cache()
class Trainer:
    def __init__(self, model, device, num_classes=5, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_val_loss = float('inf')
        
        self.class_to_rgb = {
            0: (155, 38, 182),    # obstacles
            1: (14, 135, 204),    # water
            2: (124, 252, 0),     # soft-surfaces
            3: (255, 20, 147),    # moving-objects
            4: (169, 169, 169)    # landing-zones
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs):
        print("Starting training...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved best model with validation loss: {val_loss:.4f}")

    def predict(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_idx, rgb in self.class_to_rgb.items():
            rgb_mask[mask == class_idx] = rgb
        
        rgb_mask = Image.fromarray(rgb_mask).resize(original_size, Image.NEAREST)
        
        return rgb_mask

def get_data_loaders(image_dir, mask_dir, batch_size=8, val_split=0.2, image_size=512):
    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    mask_paths = sorted([
        os.path.join(mask_dir, fname)
        for fname in os.listdir(mask_dir)
        if fname.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SegmentationDataset(
        train_img_paths,
        train_mask_paths,
        transform=train_transform
    )
    
    val_dataset = SegmentationDataset(
        val_img_paths,
        val_mask_paths,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNET(in_channels=3, out_channels=5)
    trainer = Trainer(model, device)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        image_dir="/home/cha0s/Desktop/uas-tasks/MultiClassSegmentation/classes_dataset/original_images",
        mask_dir="/home/cha0s/Desktop/uas-tasks/MultiClassSegmentation/classes_dataset/label_images_semantic",
        batch_size=2
    )
    
    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=50)