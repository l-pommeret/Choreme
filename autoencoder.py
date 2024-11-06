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
import gc

# Optimisation de la mémoire CUDA
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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
        hidden_dim = dim
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + 0.1 * self.layers(x)  # Scaled residual connection

class EnhancedDenseEncoder(nn.Module):
    def __init__(self, input_dim, scale_latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Dimensions réduites
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, scale_latent_dim)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(512),
            nn.LayerNorm(scale_latent_dim)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(1024),
            ResidualBlock(512),
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
        
        # Dimensions réduites
        self.decoder_layers = nn.ModuleList([
            nn.Linear(latent_dim, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, output_dim)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(512),
            nn.LayerNorm(1024),
            nn.LayerNorm(output_dim)
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512),
            ResidualBlock(1024),
            ResidualBlock(output_dim)
        ])
        
        # Fusion layers for skip connections
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim * 2, dim) for dim in [512, 1024, output_dim]
        ])
        
    def forward(self, x, encoder_features):
        for i, (layer, norm, res) in enumerate(zip(self.decoder_layers, self.norms, self.residual_blocks)):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = res(x)
            
            if i < len(encoder_features):
                fusion_input = torch.cat([x, encoder_features[-(i+1)]], dim=1)
                x = self.fusion_layers[i](fusion_input)
        
        return torch.sigmoid(x.view(-1, *self.output_shape))

class LightPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()  # Utilise seulement les premières couches
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, target):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        x = (x - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        perceptual_loss = 0
        style_loss = 0
        
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
            
        return perceptual_loss, style_loss

class EnhancedMultiScaleAutoencoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), scale_latent_dim=256, final_latent_dim=128):
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        encoder_features_dims = [1024, 512, scale_latent_dim]
        
        self.micro_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        self.meso_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        self.macro_encoder = EnhancedDenseEncoder(input_dim, scale_latent_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=scale_latent_dim,
            num_heads=4,  # Réduit de 8 à 4
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

def train_model(data_dir, num_epochs=50, batch_size=32):  # Batch size réduit à 32
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
    perceptual_loss_fn = LightPerceptualLoss().to(device)
    
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
                perceptual_loss, style_loss = perceptual_loss_fn(outputs[0], micro)
                
                # Calcul des pertes de reconstruction
                recon_loss = (
                    F.l1_loss(outputs[0], micro) * scale_weights['micro'] +
                    F.l1_loss(outputs[1], meso) * scale_weights['meso'] +
                    F.l1_loss(outputs[2], macro) * scale_weights['macro']
                )
                
                # Perte totale
                loss = recon_loss + 0.1 * perceptual_loss + style_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            # Libération de la mémoire
            del outputs, loss, perceptual_loss, style_loss
            torch.cuda.empty_cache()
        
        avg_loss = np.mean(epoch_losses)
        print(f'\nEpoch {epoch+1}: Loss = {avg_loss:.6f}')
        
        # Sauvegarde du modèle
        if avg_loss < best_test_loss:
            best_test_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Visualisation des reconstructions
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_micro, test_meso, test_macro = next(iter(test_loader))
                test_micro = test_micro.to(device)
                recon_micro, _, _, _ = model(test_micro, test_meso.to(device), test_macro.to(device))
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(test_micro[0].cpu().numpy().transpose(1, 2, 0))
                plt.title('Original')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(recon_micro[0].cpu().numpy().transpose(1, 2, 0))
                plt.title('Reconstructed')
                plt.axis('off')
                
                plt.savefig(f'reconstruction_epoch_{epoch+1}.png')
                plt.close()
                
                # Libération de la mémoire
                del recon_micro
                torch.cuda.empty_cache()
        
        # Nettoyage de la mémoire à la fin de chaque époque
        gc.collect()
        torch.cuda.empty_cache()
    
    return model, train_losses

if __name__ == "__main__":
    # Configuration avec les paramètres optimisés pour la mémoire
    config = {
        "data_dir": "data",
        "num_epochs": 50,
        "batch_size": 16  # Batch size encore plus petit si nécessaire
    }
    
    # Nettoyage initial de la mémoire
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        model, train_losses = train_model(**config)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM détecté. Essayons avec un batch size plus petit...")
            config["batch_size"] = 8
            model, train_losses = train_model(**config)
        else:
            raise e