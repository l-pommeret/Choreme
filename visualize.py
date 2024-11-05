import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from autoencoder import MultiScaleAutoencoder, MultiScaleImageDataset
from torch.utils.data import DataLoader

def load_checkpoint(model, checkpoint_path):
    """Charge un modèle sauvegardé"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['test_loss']

def visualize_random_samples(model, test_loader, device, num_samples=5):
    """Visualise des échantillons aléatoires et leurs reconstructions"""
    model.eval()
    
    # Récupérer tous les indices valides
    dataset_size = len(test_loader.dataset)
    indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 4*num_samples))
        fig.suptitle('Comparaison Original vs Reconstruction', fontsize=16)
        
        # Titres des colonnes
        cols = ['Micro Original', 'Micro Reconst.', 
                'Meso Original', 'Meso Reconst.',
                'Macro Original', 'Macro Reconst.']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        
        for i, idx in enumerate(indices):
            # Charger une image
            micro, meso, macro = test_loader.dataset[idx]
            
            # Ajouter la dimension batch
            micro = micro.unsqueeze(0).to(device)
            meso = meso.unsqueeze(0).to(device)
            macro = macro.unsqueeze(0).to(device)
            
            # Obtenir les reconstructions
            micro_out, meso_out, macro_out, latent = model(micro, meso, macro)
            
            # Afficher les originaux et reconstructions
            # Micro
            axes[i, 0].imshow(micro.cpu().squeeze())  # Retrait de cmap='gray'
            axes[i, 1].imshow(micro_out.cpu().squeeze())

            # Meso
            axes[i, 2].imshow(meso.cpu().squeeze())
            axes[i, 3].imshow(meso_out.cpu().squeeze())

            # Macro
            axes[i, 4].imshow(macro.cpu().squeeze())
            axes[i, 5].imshow(macro_out.cpu().squeeze())
            
            # Retirer les axes
            for ax in axes[i]:
                ax.axis('off')
                
            # Ajouter les erreurs de reconstruction
            micro_error = torch.nn.functional.mse_loss(micro, micro_out).item()
            meso_error = torch.nn.functional.mse_loss(meso, meso_out).item()
            macro_error = torch.nn.functional.mse_loss(macro, macro_out).item()
            
            # Ajouter un titre à la ligne avec les erreurs
            axes[i, 0].set_ylabel(f'Sample {i+1}\nMicro Err: {micro_error:.4f}\nMeso Err: {meso_error:.4f}\nMacro Err: {macro_error:.4f}')
        
        plt.tight_layout()
        plt.savefig('random_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_latent_space(model, test_loader, device, n_components=2):
    """Visualise l'espace latent en 2D avec PCA"""
    from sklearn.decomposition import PCA
    
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for micro, meso, macro in test_loader:
            micro = micro.to(device)
            meso = meso.to(device)
            macro = macro.to(device)
            
            _, _, _, latent = model(micro, meso, macro)
            latent_vectors.append(latent.cpu().numpy())
    
    # Concatener tous les vecteurs latents
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Appliquer PCA
    pca = PCA(n_components=n_components)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Visualiser
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    plt.title('Projection 2D de l\'espace latent (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
    plt.savefig('latent_space.png')
    plt.show()

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"  # Le dossier contenant les sous-dossiers train et test
    checkpoint_path = "best_model.pth"  # Le chemin vers le meilleur modèle sauvegardé
    
    # Charger le modèle
    model = MultiScaleAutoencoder().to(device)
    model, epoch, test_loss = load_checkpoint(model, checkpoint_path)
    print(f"Modèle chargé de l'époque {epoch} avec loss de test {test_loss:.6f}")
    
    # Charger le dataset de test
    test_dataset = MultiScaleImageDataset(data_dir, 'test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Visualiser des échantillons aléatoires
    print("Génération des visualisations des reconstructions...")
    visualize_random_samples(model, test_loader, device, num_samples=5)
    
    # Visualiser l'espace latent
    print("Génération de la visualisation de l'espace latent...")
    visualize_latent_space(model, test_loader, device)

if __name__ == "__main__":
    main()