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
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['test_loss']

def prepare_image_for_display(tensor):
    """Prépare un tenseur RGB pour l'affichage"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)

def visualize_random_samples(model, test_loader, device, num_samples=5):
    """Visualise des échantillons aléatoires et leurs reconstructions en RGB"""
    model.eval()
    
    dataset_size = len(test_loader.dataset)
    indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 4*num_samples))
        fig.suptitle('Comparaison Original vs Reconstruction (RGB)', fontsize=16)
        
        cols = ['Micro Original', 'Micro Reconst.', 
                'Meso Original', 'Meso Reconst.',
                'Macro Original', 'Macro Reconst.']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        
        for i, idx in enumerate(indices):
            # Chargement des images
            micro, meso, macro = test_loader.dataset[idx]
            
            # Préparation pour le modèle
            micro = micro.unsqueeze(0).to(device)
            meso = meso.unsqueeze(0).to(device)
            macro = macro.unsqueeze(0).to(device)
            
            # Génération des reconstructions
            micro_out, meso_out, macro_out, latent = model(micro, meso, macro)
            
            # Préparation pour l'affichage
            pairs = [
                (micro.squeeze(), micro_out.squeeze()),
                (meso.squeeze(), meso_out.squeeze()),
                (macro.squeeze(), macro_out.squeeze())
            ]
            
            # Affichage des images
            for j, (orig, recon) in enumerate(pairs):
                axes[i, j*2].imshow(prepare_image_for_display(orig))
                axes[i, j*2+1].imshow(prepare_image_for_display(recon))
                axes[i, j*2].axis('off')
                axes[i, j*2+1].axis('off')
            
            # Calcul et affichage des erreurs
            errors = {
                'micro': torch.nn.functional.mse_loss(micro, micro_out).item(),
                'meso': torch.nn.functional.mse_loss(meso, meso_out).item(),
                'macro': torch.nn.functional.mse_loss(macro, macro_out).item()
            }
            
            axes[i, 0].set_ylabel(
                f'Sample {i+1}\n' + \
                f'Micro Err: {errors["micro"]:.4f}\n' + \
                f'Meso Err: {errors["meso"]:.4f}\n' + \
                f'Macro Err: {errors["macro"]:.4f}'
            )
        
        plt.tight_layout()
        plt.savefig('random_reconstructions_rgb.png', dpi=300, bbox_inches='tight')
        plt.close()

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
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Application de PCA
    pca = PCA(n_components=n_components)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Visualisation
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=np.sum(latent_vectors**2, axis=1),
                         cmap='viridis', 
                         alpha=0.6)
    plt.colorbar(scatter, label='Norme du vecteur latent')
    plt.title('Projection 2D de l\'espace latent (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
    plt.savefig('latent_space_rgb.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de: {device}")
    
    data_dir = "data"
    checkpoint_path = "best_model_rgb.pth"  # Notez le _rgb pour être cohérent
    
    try:
        # Chargement du modèle avec input_channels=3 pour RGB
        model = MultiScaleAutoencoder(input_channels=3).to(device)
        model, epoch, test_loss = load_checkpoint(model, checkpoint_path)
        print(f"Modèle RGB chargé de l'époque {epoch} avec loss de test {test_loss:.6f}")
        
        # Chargement des données
        test_dataset = MultiScaleImageDataset(data_dir, 'test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Génération des visualisations
        print("\nGénération des reconstructions RGB...")
        visualize_random_samples(model, test_loader, device, num_samples=5)
        
        print("\nGénération de la visualisation de l'espace latent...")
        visualize_latent_space(model, test_loader, device)
        
        print("\nVisualisation terminée ! Les images ont été sauvegardées :")
        print("- random_reconstructions_rgb.png")
        print("- latent_space_rgb.png")
        
    except Exception as e:
        print(f"\nUne erreur s'est produite lors de la visualisation:")
        print(f"{str(e)}")
        print("\nAssurez-vous que:")
        print("1. Le modèle a bien été entraîné avec des images RGB (3 canaux)")
        print("2. Le fichier best_model_rgb.pth existe et correspond à un modèle RGB")
        print("3. Les images dans le dataset sont bien en RGB")
        raise

if __name__ == "__main__":
    main()