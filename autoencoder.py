import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
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

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class EnhancedDenseEncoder(nn.Module):
    def __init__(self, input_dim, scale_latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, scale_latent_dim)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(4096),
            nn.LayerNorm(2048),
            nn.LayerNorm(1024),
            nn.LayerNorm(scale_latent_dim)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(4096),
            ResidualBlock(2048),
            ResidualBlock(1024),
            ResidualBlock(scale_latent_dim)
        ])
        
    def forward(self, x):
        features = []
        x = self.flatten(x)
        
        for layer, norm, res in zip(self.encoder_layers, self.norms, self.residual_blocks):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = res(x)
            features.append(x)
            
        return x, features

class EnhancedDenseDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape, encoder_features_dims):
        super().__init__()
        self.output_shape = output_shape
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        
        self.decoder_layers = nn.ModuleList([
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, 2048),
            nn.Linear(2048, 4096),
            nn.Linear(4096, output_dim)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(2048),
            nn.LayerNorm(4096),
            nn.LayerNorm(output_dim)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(1024),
            ResidualBlock(2048),
            ResidualBlock(4096),
            ResidualBlock(output_dim)
        ])
        
        # Fusion layers for skip connections
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim * 2, dim) for dim in reversed(encoder_features_dims)
        ])
        
    def forward(self, x, encoder_features):
        for i, (layer, norm, res) in enumerate(zip(self.decoder_layers, self.norms, self.residual_blocks)):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = res(x)
            
            # Add skip connections
            if i < len(encoder_features):
                fusion_input = torch.cat([x, encoder_features[-(i+1)]], dim=1)
                x = self.fusion_layers[i](fusion_input)
        
        return torch.sigmoid(x.view(-1, *self.output_shape))

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]), # relu3_3
            nn.Sequential(*list(vgg.children())[16:23])  # relu4_3
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def forward(self, x, target):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        
        perceptual_loss = 0
        style_loss = 0
        x_features = []
        target_features = []
        
        for slice in self.slices:
            x = slice(x)
            with torch.no_grad():
                target = slice(target)
                
            perceptual_loss += F.mse_loss(x, target)
            
            # Gram matrix pour la perte de style
            b, c, h, w = x.size()
            x_flat = x.view(b, c, -1)
            target_flat = target.view(b, c, -1)
            
            x_gram = torch.bmm(x_flat, x_flat.transpose(1, 2)) / (c * h * w)
            target_gram = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (c * h * w)
            style_loss += F.mse_loss(x_gram, target_gram)
            
            x_features.append(x)
            target_features.append(target)
            
        return perceptual_loss, style_loss, x_features, target_features
    

class EnhancedMultiScaleAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), scale_latent_dim=512, final_latent_dim=256):
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        encoder_features_dims = [4096, 2048, 1024, scale_latent_dim]
        
        self.micro_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        self.meso_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        self.macro_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        
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
            ResidualBlock(final_latent_dim)
        )
        
        self.micro_decoder = EnhancedDenseDecoder(final_latent_dim, input_shape, encoder_features_dims)
        self.meso_decoder = EnhancedDenseDecoder(final_latent_dim, input_shape, encoder_features_dims)
        self.macro_decoder = EnhancedDenseDecoder(final_latent_dim, input_shape, encoder_features_dims)
        
    def forward(self, micro, meso, macro):
        micro_encoded, micro_features = self.micro_encoder(micro)
        meso_encoded, meso_features = self.meso_encoder(meso)
        macro_encoded, macro_features = self.macro_encoder(macro)
        
        encodings = torch.stack([micro_encoded, meso_encoded, macro_encoded], dim=1)
        attended_encodings, _ = self.attention(encodings, encodings, encodings)
        attended_encodings = attended_encodings.flatten(1)
        
        latent = self.fusion(attended_encodings)
        
        micro_decoded = self.micro_decoder(latent, micro_features)
        meso_decoded = self.meso_decoder(latent, meso_features)
        macro_decoded = self.macro_decoder(latent, macro_features)
        
        return micro_decoded, meso_decoded, macro_decoded, latent

def compute_multiscale_loss(outputs, targets, weights, latent, perceptual_loss_fn, 
                          content_weight=1.0, perceptual_weight=0.1, style_weight=1.0,
                          l1_lambda=1e-6, l2_lambda=1e-5):
    micro_out, meso_out, macro_out, latent = outputs
    micro_target, meso_target, macro_target = targets
    
    # Pertes de reconstruction L1
    pixel_loss_micro = F.l1_loss(micro_out, micro_target) * weights['micro']
    pixel_loss_meso = F.l1_loss(meso_out, meso_target) * weights['meso']
    pixel_loss_macro = F.l1_loss(macro_out, macro_target) * weights['macro']
    pixel_loss = pixel_loss_micro + pixel_loss_meso + pixel_loss_macro
    
    # Pertes perceptuelles et de style
    perceptual_terms = []
    style_terms = []
    for output, target in [(micro_out, micro_target), 
                          (meso_out, meso_target), 
                          (macro_out, macro_target)]:
        perc_loss, style_loss, _, _ = perceptual_loss_fn(output, target)
        perceptual_terms.append(perc_loss)
        style_terms.append(style_loss)
    
    perceptual_loss = sum(perceptual_terms)
    style_loss = sum(style_terms)
    
    # Régularisation
    reg_loss = l1_lambda * torch.abs(latent).mean() + l2_lambda * torch.square(latent).mean()
    
    # Perte totale
    total_loss = (
        content_weight * pixel_loss + 
        perceptual_weight * perceptual_loss +
        style_weight * style_loss + 
        reg_loss
    )
    
    return total_loss, {
        'total': total_loss.item(),
        'pixel': pixel_loss.item(),
        'perceptual': perceptual_loss.item(),
        'style': style_loss.item(),
        'reg': reg_loss.item(),
        'micro': pixel_loss_micro.item(),
        'meso': pixel_loss_meso.item(),
        'macro': pixel_loss_macro.item()
    }

def train_model(data_dir, num_epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    model = EnhancedMultiScaleAutoencoder().to(device)
    perceptual_loss_fn = PerceptualLoss().to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    scaler = torch.cuda.amp.GradScaler()
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    component_losses = {
        'pixel': [], 'perceptual': [], 'style': [], 
        'micro': [], 'meso': [], 'macro': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_components = {k: [] for k in component_losses.keys()}
        
        for micro, meso, macro in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            micro = micro.to(device, non_blocking=True)
            meso = meso.to(device, non_blocking=True)
            macro = macro.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(micro, meso, macro)
                loss, components = compute_multiscale_loss(
                    outputs, (micro, meso, macro),
                    scale_weights, outputs[3],
                    perceptual_loss_fn,
                    content_weight=1.0,
                    perceptual_weight=0.1,
                    style_weight=1.0
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_losses.append(components['total'])
            for k, v in components.items():
                if k in epoch_components:
                    epoch_components[k].append(v)
        
        model.eval()
        test_loss = 0
        test_components = {k: 0 for k in component_losses.keys()}
        n_test_batches = 0
        
        with torch.no_grad():
            for micro, meso, macro in test_loader:
                micro = micro.to(device, non_blocking=True)
                meso = meso.to(device, non_blocking=True)
                macro = macro.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(micro, meso, macro)
                    loss, components = compute_multiscale_loss(
                        outputs, (micro, meso, macro),
                        scale_weights, outputs[3],
                        perceptual_loss_fn
                    )
                
                test_loss += loss.item()
                for k, v in components.items():
                    if k in test_components:
                        test_components[k] += v
                n_test_batches += 1
        
        test_loss /= n_test_batches
        for k in test_components:
            test_components[k] /= n_test_batches
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        for k, v in epoch_components.items():
            component_losses[k].append(np.mean(v))
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Test Loss: {test_loss:.6f}')
        print('\nComponent Losses:')
        for k, v in epoch_components.items():
            print(f'  {k}: {np.mean(v):.6f}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Sauvegarder le meilleur modèle
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f'Saving new best model (test loss: {test_loss:.6f})')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
                'component_losses': component_losses
            }, 'best_model.pth')
        
        # Visualiser les progrès
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plt.figure(figsize=(15, 5))
            
            # Plot des pertes totales
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(test_losses, label='Test')
            plt.title('Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot des composantes de perte
            plt.subplot(1, 3, 2)
            for k, v in component_losses.items():
                if k not in ['micro', 'meso', 'macro']:
                    plt.plot(v, label=k)
            plt.title('Loss Components')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot des pertes par échelle
            plt.subplot(1, 3, 3)
            for k in ['micro', 'meso', 'macro']:
                plt.plot(component_losses[k], label=k)
            plt.title('Scale Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'training_progress_epoch_{epoch+1}.png')
            plt.close()
            
            # Sauvegarder quelques exemples de reconstruction
            with torch.no_grad():
                model.eval()
                test_micro, test_meso, test_macro = next(iter(test_loader))
                test_micro = test_micro.to(device)
                test_meso = test_meso.to(device)
                test_macro = test_macro.to(device)
                
                recon_micro, recon_meso, recon_macro, _ = model(test_micro, test_meso, test_macro)
                
                def save_image_grid(original, reconstructed, name, idx=0):
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(original[idx].cpu().numpy().transpose(1, 2, 0))
                    plt.title('Original')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(reconstructed[idx].cpu().numpy().transpose(1, 2, 0))
                    plt.title('Reconstructed')
                    plt.axis('off')
                    
                    plt.savefig(f'reconstruction_{name}_epoch_{epoch+1}.png')
                    plt.close()
                
                save_image_grid(test_micro, recon_micro, 'micro')
                save_image_grid(test_meso, recon_meso, 'meso')
                save_image_grid(test_macro, recon_macro, 'macro')
    
    return model, train_losses, test_losses, component_losses

if __name__ == "__main__":
    config = {
        "data_dir": "data",
        "num_epochs": 50,
        "batch_size": 64
    }
    
    model, train_losses, test_losses, component_losses = train_model(**config)