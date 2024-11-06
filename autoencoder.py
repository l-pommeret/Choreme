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

class DenseEncoder(nn.Module):
    def __init__(self, input_dim, scale_latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, scale_latent_dim),
            nn.LayerNorm(scale_latent_dim)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.encoder(x)

class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, *self.output_shape)

class MultiScaleDenseAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), scale_latent_dim=512, final_latent_dim=256):
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.micro_encoder = DenseEncoder(input_dim, scale_latent_dim)
        self.meso_encoder = DenseEncoder(input_dim, scale_latent_dim)
        self.macro_encoder = DenseEncoder(input_dim, scale_latent_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=scale_latent_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(scale_latent_dim * 3, final_latent_dim),
            nn.LayerNorm(final_latent_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.micro_decoder = DenseDecoder(final_latent_dim, input_shape)
        self.meso_decoder = DenseDecoder(final_latent_dim, input_shape)
        self.macro_decoder = DenseDecoder(final_latent_dim, input_shape)
        
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def compute_loss(outputs, targets, weights, latent, l1_lambda=1e-6, l2_lambda=1e-5):
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
        'macro': macro_loss.item(),
        'reg': (l1_loss + l2_loss).item()
    }

def train_model(data_dir, num_epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Augmentation du poids des échelles meso et macro
    scale_weights = {
        'micro': 1.0,
        'meso': 0.8,  
        'macro': 0.6
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
    
    model = MultiScaleDenseAutoencoder().to(device)
    model.apply(init_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-3,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-2,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    scaler = torch.cuda.amp.GradScaler()  # Pour mixed precision training
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
            
            with torch.cuda.amp.autocast():
                outputs = model(micro, meso, macro)
                loss, loss_components = compute_loss(
                    outputs, (micro, meso, macro), 
                    scale_weights, outputs[3]
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_losses.append(loss_components)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for micro, meso, macro in test_loader:
                micro = micro.to(device, non_blocking=True)
                meso = meso.to(device, non_blocking=True)
                macro = macro.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(micro, meso, macro)
                    loss, _ = compute_loss(outputs, (micro, meso, macro), scale_weights, outputs[3])
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        avg_train_loss = np.mean([l['total'] for l in epoch_losses])
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        components = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
        for k, v in components.items():
            print(f'  {k}: {v:.6f}')
        print(f'Test Loss: {test_loss:.6f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
            }, 'best_model.pth')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(test_losses, label='Test')
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            for k in ['reconstruction', 'micro', 'meso', 'macro', 'reg']:
                values = [l[k] for l in epoch_losses]
                plt.plot(values, label=k)
            plt.title('Loss Components')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'training_progress_epoch_{epoch+1}.png')
            plt.close()
    
    return model, train_losses, test_losses

if __name__ == "__main__":
    config = {
        "data_dir": "data",
        "num_epochs": 50,
        "batch_size": 64
    }
    
    model, train_losses, test_losses = train_model(**config)