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
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        identity = x
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return x + identity

class ConvEncoder(nn.Module):
    def __init__(self, scale_latent_dim):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, scale_latent_dim),
            nn.LayerNorm(scale_latent_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 128 * 8 * 8),
            nn.LayerNorm(128 * 8 * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.conv1 = ConvBlock(128, 64)
        self.conv2 = ConvBlock(64, 32)
        self.conv3 = ConvBlock(32, 16)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3(x)
        
        x = self.final_conv(x)
        return torch.sigmoid(x)

class MultiScaleCNNAutoencoder(nn.Module):
    def __init__(self, scale_latent_dim=128, final_latent_dim=96):
        super().__init__()
        
        self.micro_encoder = ConvEncoder(scale_latent_dim)
        self.meso_encoder = ConvEncoder(scale_latent_dim)
        self.macro_encoder = ConvEncoder(scale_latent_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=scale_latent_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(scale_latent_dim * 3, final_latent_dim),
            nn.LayerNorm(final_latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.micro_decoder = ConvDecoder(final_latent_dim)
        self.meso_decoder = ConvDecoder(final_latent_dim)
        self.macro_decoder = ConvDecoder(final_latent_dim)
        
    def forward(self, micro, meso, macro):
        micro_encoded = self.micro_encoder(micro)
        meso_encoded = self.meso_encoder(meso)
        macro_encoded = self.macro_encoder(macro)
        
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
    
    mse_criterion = nn.MSELoss()
    
    micro_loss = mse_criterion(micro_out, micro_target) * weights['micro']
    meso_loss = mse_criterion(meso_out, meso_target) * weights['meso']
    macro_loss = mse_criterion(macro_out, macro_target) * weights['macro']
    
    reconstruction_loss = micro_loss + meso_loss + macro_loss
    
    l1_loss = l1_lambda * torch.abs(latent).mean()
    l2_loss = l2_lambda * torch.square(latent).mean()
    
    total_loss = reconstruction_loss + l1_loss + l2_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'reconstruction': reconstruction_loss.item(),
        'micro': micro_loss.item(),
        'meso': meso_loss.item(),
        'macro': macro_loss.item()
    }

def train_model(data_dir, num_epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    scale_weights = {
        'micro': 0.8,
        'meso': 0.1,
        'macro': 0.1
    }
    
    train_dataset = MultiScaleImageDataset(data_dir, 'train')
    test_dataset = MultiScaleImageDataset(data_dir, 'test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = MultiScaleCNNAutoencoder().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for micro, meso, macro in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            micro = micro.to(device, non_blocking=True)
            meso = meso.to(device, non_blocking=True)
            macro = macro.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(micro, meso, macro)
            
            loss, loss_components = compute_loss(
                outputs, (micro, meso, macro), 
                scale_weights, outputs[3]
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for micro, meso, macro in test_loader:
                micro = micro.to(device, non_blocking=True)
                meso = meso.to(device, non_blocking=True)
                macro = macro.to(device, non_blocking=True)
                
                outputs = model(micro, meso, macro)
                loss, _ = compute_loss(outputs, (micro, meso, macro), scale_weights, outputs[3])
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        avg_train_loss = np.mean(epoch_losses)
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Test Loss: {test_loss:.6f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Sauvegarde uniquement si amélioration significative
        if test_loss < best_test_loss * 0.95:
            best_test_loss = test_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_loss': test_loss,
            }, 'best_model.pth')
        
        # Plot moins fréquent
        if (epoch + 1) % 10 == 0 or epoch == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train')
            plt.plot(test_losses, label='Test')
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('training_progress.png')
            plt.close()
    
    return model, train_losses, test_losses

if __name__ == "__main__":
    config = {
        "data_dir": "data",
        "num_epochs": 50,
        "batch_size": 64
    }
    
    model, train_losses, test_losses = train_model(**config)