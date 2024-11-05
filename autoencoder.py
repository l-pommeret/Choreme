import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Dataset personnalisé pour charger les images multi-échelles
class MultiScaleImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.locations = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, idx):
        loc_dir = os.path.join(self.dataset_dir, self.locations[idx])
        
        # Charger les trois échelles
        micro = self.load_image(os.path.join(loc_dir, 'micro.png'))
        meso = self.load_image(os.path.join(loc_dir, 'meso.png'))
        macro = self.load_image(os.path.join(loc_dir, 'macro.png'))
        
        return micro, meso, macro
    
    def load_image(self, path):
        img = Image.open(path).convert('L')  # Conversion en niveaux de gris
        img_array = np.array(img) / 255.0  # Normalisation
        return torch.FloatTensor(img_array)

class MultiScaleAutoencoder(nn.Module):
    def __init__(self, scale_latent_dim=64, final_latent_dim=128):
        super(MultiScaleAutoencoder, self).__init__()
        
        # Encodeurs pour chaque échelle
        self.micro_encoder = self.create_encoder(scale_latent_dim)
        self.meso_encoder = self.create_encoder(scale_latent_dim)
        self.macro_encoder = self.create_encoder(scale_latent_dim)
        
        # Fusion et projection vers l'espace latent final
        self.fusion = nn.Sequential(
            nn.Linear(scale_latent_dim * 3, final_latent_dim),
            nn.ReLU()
        )
        
        # Décodeurs pour chaque échelle
        self.micro_decoder = self.create_decoder()
        self.meso_decoder = self.create_decoder()
        self.macro_decoder = self.create_decoder()
        
    def create_encoder(self, latent_dim):
        return nn.Sequential(
            nn.Flatten(),  # 64*64 -> 4096
            nn.Linear(64*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
    
    def create_decoder(self):
        return nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*64),
            nn.Sigmoid()
        )
    
    def forward(self, micro, meso, macro):
        # Encodage de chaque échelle
        micro_encoded = self.micro_encoder(micro)
        meso_encoded = self.meso_encoder(meso)
        macro_encoded = self.macro_encoder(macro)
        
        # Fusion des représentations
        combined = torch.cat([micro_encoded, meso_encoded, macro_encoded], dim=1)
        latent = self.fusion(combined)
        
        # Décodage pour chaque échelle
        micro_decoded = self.micro_decoder(latent).view(-1, 64, 64)
        meso_decoded = self.meso_decoder(latent).view(-1, 64, 64)
        macro_decoded = self.macro_decoder(latent).view(-1, 64, 64)
        
        return micro_decoded, meso_decoded, macro_decoded

def train_model(dataset_dir, num_epochs=50):
    # Paramètres
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    
    # Pondération des échelles dans la loss
    scale_weights = {
        'micro': 0.5,  # 50% du poids pour l'échelle micro
        'meso': 0.3,   # 30% pour l'échelle meso
        'macro': 0.2   # 20% pour l'échelle macro
    }
    
    # Préparation des données
    dataset = MultiScaleImageDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modèle et optimiseur
    model = MultiScaleAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for micro, meso, macro in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Passage des données au device
            micro, meso, macro = micro.to(device), meso.to(device), macro.to(device)
            
            # Forward pass
            micro_out, meso_out, macro_out = model(micro, meso, macro)
            
            # Calcul des pertes pour chaque échelle
            micro_loss = criterion(micro_out, micro) * scale_weights['micro']
            meso_loss = criterion(meso_out, meso) * scale_weights['meso']
            macro_loss = criterion(macro_out, macro) * scale_weights['macro']
            
            # Perte totale pondérée
            loss = micro_loss + meso_loss + macro_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Affichage des métriques
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
        
        # Sauvegarde du modèle
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'multiscale_autoencoder_epoch{epoch+1}.pth')
    
    return model

if __name__ == "__main__":
    trained_model = train_model("dataset", num_epochs=50)