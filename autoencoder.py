import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class MultiScaleImageDataset(Dataset):
    def __init__(self, data_dir, subset='train'):
        self.data_dir = os.path.join(data_dir, subset)
        self.locations = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        loc_dir = os.path.join(self.data_dir, self.locations[idx])
        
        # Charger les trois échelles
        micro = self.load_image(os.path.join(loc_dir, 'micro.png'))
        meso = self.load_image(os.path.join(loc_dir, 'meso.png'))
        macro = self.load_image(os.path.join(loc_dir, 'macro.png'))
        
        return micro, meso, macro
    
    def load_image(self, path):
        """Charge une image RGB et la normalise"""
        img = Image.open(path)
        img_array = np.array(img).transpose(2, 0, 1) / 255.0  # Format CHW pour PyTorch
        return torch.FloatTensor(img_array)

class MultiScaleAutoencoder(nn.Module):
    def __init__(self, scale_latent_dim=64, final_latent_dim=128):
        super(MultiScaleAutoencoder, self).__init__()
        
        # On fixe directement input_dim pour RGB (3 canaux)
        self.input_dim = 3 * 64 * 64  # 3 canaux RGB
        self.scale_latent_dim = scale_latent_dim
        self.final_latent_dim = final_latent_dim
        
        # Encodeurs pour chaque échelle
        self.micro_encoder = self.create_encoder(scale_latent_dim)
        self.meso_encoder = self.create_encoder(scale_latent_dim)
        self.macro_encoder = self.create_encoder(scale_latent_dim)
        
        # Fusion avec BatchNorm
        self.fusion = nn.Sequential(
            nn.Linear(scale_latent_dim * 3, final_latent_dim),
            nn.BatchNorm1d(final_latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Décodeurs pour chaque échelle
        self.micro_decoder = self.create_decoder()
        self.meso_decoder = self.create_decoder()
        self.macro_decoder = self.create_decoder()
        
    def create_encoder(self, latent_dim):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 1024),  # Utilise self.input_dim au lieu de 64*64
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
    
    def create_decoder(self):
        return nn.Sequential(
            nn.Linear(self.final_latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),  # Utilise self.input_dim au lieu de 64*64
            nn.Sigmoid()
        )
    
    def forward(self, micro, meso, macro):
        micro_encoded = self.micro_encoder(micro)
        meso_encoded = self.meso_encoder(meso)
        macro_encoded = self.macro_encoder(macro)
        
        combined = torch.cat([micro_encoded, meso_encoded, macro_encoded], dim=1)
        latent = self.fusion(combined)
        
        micro_decoded = self.micro_decoder(latent).view(-1, 3, 64, 64)  # 3 canaux
        meso_decoded = self.meso_decoder(latent).view(-1, 3, 64, 64)   # 3 canaux
        macro_decoded = self.macro_decoder(latent).view(-1, 3, 64, 64)  # 3 canaux
        
        return micro_decoded, meso_decoded, macro_decoded, latent

# Dans train_model, changer cette ligne :
def train_model(data_dir, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de {device}")
    
    # Paramètres
    batch_size = 32
    scale_weights = {
        'micro': 0.5,
        'meso': 0.3,
        'macro': 0.2
    }
    
    # Chargement des données
    train_dataset = MultiScaleImageDataset(data_dir, 'train')
    test_dataset = MultiScaleImageDataset(data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Modèle et optimisation
    model = MultiScaleAutoencoder().to(device)  # Retrait de input_channels=3
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Boucle d'entraînement
        for micro, meso, macro in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            micro, meso, macro = micro.to(device), meso.to(device), macro.to(device)
            
            optimizer.zero_grad()
            outputs = model(micro, meso, macro)
            
            loss, loss_components = compute_loss(outputs, (micro, meso, macro), 
                                               scale_weights, outputs[3])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation
        avg_train_loss = total_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, scale_weights, device)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Test Loss: {avg_test_loss:.6f}')
        print(f'Loss Components: {loss_components}')
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Sauvegarde du meilleur modèle
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, 'best_model_rgb.pth')
    
    return model, train_losses, test_losses

def compute_loss(outputs, targets, weights, latent, l1_lambda=1e-5, l2_lambda=1e-4):
    micro_out, meso_out, macro_out, latent = outputs
    micro_target, meso_target, macro_target = targets
    
    criterion = nn.MSELoss()
    
    # Pertes de reconstruction pondérées
    micro_loss = criterion(micro_out, micro_target) * weights['micro']
    meso_loss = criterion(meso_out, meso_target) * weights['meso']
    macro_loss = criterion(macro_out, macro_target) * weights['macro']
    
    reconstruction_loss = micro_loss + meso_loss + macro_loss
    
    # Régularisation
    l1_loss = l1_lambda * torch.abs(latent).mean()
    l2_loss = l2_lambda * torch.square(latent).mean()
    
    return reconstruction_loss + l1_loss + l2_loss, {
        'reconstruction': reconstruction_loss.item(),
        'l1': l1_loss.item(),
        'l2': l2_loss.item()
    }

def evaluate_model(model, dataloader, weights, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for micro, meso, macro in dataloader:
            micro, meso, macro = micro.to(device), meso.to(device), macro.to(device)
            outputs = model(micro, meso, macro)
            loss, _ = compute_loss(outputs, (micro, meso, macro), weights, outputs[3])
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    model, train_losses, test_losses = train_model("data", num_epochs=300)