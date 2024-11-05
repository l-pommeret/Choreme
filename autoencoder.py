import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class MultiScaleImageDataset(Dataset):
    def __init__(self, data_dir, subset='train', transform=None):
        self.data_dir = os.path.join(data_dir, subset)
        self.locations = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        self.transform = transform
        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        loc_dir = os.path.join(self.data_dir, self.locations[idx])
        
        micro = self.load_image(os.path.join(loc_dir, 'micro.png'))
        meso = self.load_image(os.path.join(loc_dir, 'meso.png'))
        macro = self.load_image(os.path.join(loc_dir, 'macro.png'))
        
        if self.transform:
            micro = self.transform(micro)
            meso = self.transform(meso)
            macro = self.transform(macro)
        
        return micro, meso, macro
    
    def load_image(self, path):
        img = Image.open(path)
        img_array = np.array(img).transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(img_array)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        return x + identity  # Skip connection

class ConvEncoder(nn.Module):
    def __init__(self, scale_latent_dim):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, scale_latent_dim),
            nn.BatchNorm1d(scale_latent_dim)
        )
        
    def forward(self, x):
        # Encoder path with skip connections
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU()
        )
        
        self.conv1 = ConvBlock(256, 128)
        self.conv2 = ConvBlock(128, 64)
        self.conv3 = ConvBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        
        x = self.final_conv(x)
        return torch.sigmoid(x)

class MultiScaleCNNAutoencoder(nn.Module):
    def __init__(self, scale_latent_dim=512, final_latent_dim=512):
        super().__init__()
        
        self.micro_encoder = ConvEncoder(scale_latent_dim)
        self.meso_encoder = ConvEncoder(scale_latent_dim)
        self.macro_encoder = ConvEncoder(scale_latent_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=scale_latent_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(scale_latent_dim * 3, final_latent_dim),
            nn.BatchNorm1d(final_latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.micro_decoder = ConvDecoder(final_latent_dim)
        self.meso_decoder = ConvDecoder(final_latent_dim)
        self.macro_decoder = ConvDecoder(final_latent_dim)
        
    def forward(self, micro, meso, macro):
        micro_encoded = self.micro_encoder(micro)
        meso_encoded = self.meso_encoder(meso)
        macro_encoded = self.macro_encoder(macro)
        
        # Attention mechanism
        encodings = torch.stack([micro_encoded, meso_encoded, macro_encoded], dim=1)
        attended_encodings, _ = self.attention(encodings, encodings, encodings)
        attended_encodings = attended_encodings.flatten(1)
        
        latent = self.fusion(attended_encodings)
        
        micro_decoded = self.micro_decoder(latent)
        meso_decoded = self.meso_decoder(latent)
        macro_decoded = self.macro_decoder(latent)
        
        return micro_decoded, meso_decoded, macro_decoded, latent

def compute_loss(outputs, targets, weights, latent, l1_lambda=1e-5, l2_lambda=1e-4):
    micro_out, meso_out, macro_out, latent = outputs
    micro_target, meso_target, macro_target = targets
    
    # Reconstruction loss avec SSIM
    def ssim_loss(pred, target):
        return 1 - torch.mean(pytorch_ssim.ssim(pred, target))
    
    # Combine MSE et SSIM
    mse_criterion = nn.MSELoss()
    
    def combined_loss(pred, target, weight):
        mse = mse_criterion(pred, target)
        return weight * mse
    
    micro_loss = combined_loss(micro_out, micro_target, weights['micro'])
    meso_loss = combined_loss(meso_out, meso_target, weights['meso'])
    macro_loss = combined_loss(macro_out, macro_target, weights['macro'])
    
    reconstruction_loss = micro_loss + meso_loss + macro_loss
    
    # Regularization
    l1_loss = l1_lambda * torch.abs(latent).mean()
    l2_loss = l2_lambda * torch.square(latent).mean()
    
    total_loss = reconstruction_loss + l1_loss + l2_loss
    
    loss_components = {
        'reconstruction': reconstruction_loss.item(),
        'micro_loss': micro_loss.item(),
        'meso_loss': meso_loss.item(),
        'macro_loss': macro_loss.item(),
        'l1': l1_loss.item(),
        'l2': l2_loss.item()
    }
    
    return total_loss, loss_components

def train_model(data_dir, num_epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Scale weights
    scale_weights = {
        'micro': 0.5,
        'meso': 0.3,
        'macro': 0.2
    }
    
    # Data loading
    train_dataset = MultiScaleImageDataset(data_dir, 'train')
    test_dataset = MultiScaleImageDataset(data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Model initialization
    model = MultiScaleCNNAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Training tracking
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for micro, meso, macro in progress_bar:
            micro, meso, macro = micro.to(device), meso.to(device), macro.to(device)
            
            optimizer.zero_grad()
            outputs = model(micro, meso, macro)
            
            loss, loss_components = compute_loss(
                outputs, (micro, meso, macro), 
                scale_weights, outputs[3]
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Evaluation phase
        model.eval()
        test_loss = evaluate_model(model, test_loader, scale_weights, device)
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Test Loss: {test_loss:.6f}')
        print('Loss Components:', loss_components)
        
        # Learning rate adjustment
        scheduler.step(test_loss)
        
        # Model checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
            }, 'best_model_cnn.pth')
            
        # Plot training curves
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'training_progress_epoch_{epoch+1}.png')
            plt.close()
    
    return model, train_losses, test_losses

def evaluate_model(model, dataloader, weights, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for micro, meso, macro in dataloader:
            micro = micro.to(device)
            meso = meso.to(device)
            macro = macro.to(device)
            
            outputs = model(micro, meso, macro)
            loss, _ = compute_loss(outputs, (micro, meso, macro), weights, outputs[3])
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Training configuration
    config = {
        "data_dir": "data",
        "num_epochs": 300,
        "batch_size": 32
    }
    
    # Train the model
    model, train_losses, test_losses = train_model(**config)