import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from autoencoder import MultiScaleAutoencoder, MultiScaleImageDataset
from torch.utils.data import DataLoader

def load_checkpoint(model, checkpoint_path):
    """Charge un modèle sauvegardé avec gestion des erreurs"""
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Sécurité améliorée
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['test_loss']

def prepare_image_for_display(tensor):
    """Prépare un tenseur pour l'affichage en tant qu'image RGB"""
    # Conversion de (C,H,W) à (H,W,C) et passage sur CPU
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    # Normalisation dans [0,1]
    img = np.clip(img, 0, 1)
    return img

def visualize_random_samples(model, test_loader, device, num_samples=5):
    """Visualise des échantillons aléatoires et leurs reconstructions en RGB"""
    model.eval()
    
    dataset_size = len(test_loader.dataset)
    indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 4*num_samples))
        fig.suptitle('Comparaison Original vs Reconstruction (RGB)', fontsize=16)
        
        # Configuration des titres
        cols = ['Micro Original', 'Micro Reconst.', 
                'Meso Original', 'Meso Reconst.',
                'Macro Original', 'Macro Reconst.']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        
        # Traitement de chaque échantillon
        for i, idx in enumerate(indices):
            # Chargement des données
            micro, meso, macro = test_loader.dataset[idx]
            
            # Préparation pour le modèle
            micro = micro.unsqueeze(0).to(device)
            meso = meso.unsqueeze(0).to(device)
            macro = macro.unsqueeze(0).to(device)
            
            # Génération des reconstructions
            micro_out, meso_out, macro_out, latent = model(micro, meso, macro)
            
            # Préparation des images pour l'affichage
            micro_disp = prepare_image_for_display(micro.squeeze())
            micro_out_disp = prepare_image_for_display(micro_out.squeeze())
            meso_disp = prepare_image_for_display(meso.squeeze())
            meso_out_disp = prepare_image_for_display(meso_out.squeeze())
            macro_disp = prepare_image_for_display(macro.squeeze())
            macro_out_disp = prepare_image_for_display(macro_out.squeeze())
            
            # Affichage des images
            axes[i, 0].imshow(micro_disp)
            axes[i, 1].imshow(micro_out_disp)
            axes[i, 2].imshow(meso_disp)
            axes[i, 3].imshow(meso_out_disp)
            axes[i, 4].imshow(macro_disp)
            axes[i, 5].imshow(macro_out_disp)
            
            # Configuration des axes
            for ax in axes[i]:
                ax.axis('off')
            
            # Calcul et affichage des erreurs
            errors = {
                'micro': torch.nn.functional.mse_loss(micro, micro_out).item(),
                'meso': torch.nn.functional.mse_loss(meso, meso_out).item(),
                'macro': torch.nn.functional.mse_loss(macro, macro_out).item()
            }
            
            # Ajout du titre avec les erreurs
            error_text = f'Sample {i+1}\n' + \
                        f'Micro Err: {errors["micro"]:.4f}\n' + \
                        f'Meso Err: {errors["meso"]:.4f}\n' + \
                        f'Macro Err: {errors["macro"]:.4f}'
            axes[i, 0].set_ylabel(error_text)
        
        plt.tight_layout()
        plt.savefig('random_reconstructions_rgb.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_latent_space(model, test_loader, device, n_components=2):
    """Visualise l'espace latent en 2D avec PCA et coloration améliorée"""
    from sklearn.decomposition import PCA
    
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for micro, meso, macro in test_loader:
            micro, meso, macro = micro.to(device), meso.to(device), macro.to(device)
            _, _, _, latent = model(micro, meso, macro)
            latent_vectors.append(latent.cpu().numpy())
    
    # Préparation des données
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Application de PCA
    pca = PCA(n_components=n_components)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Visualisation améliorée
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=np.sum(latent_vectors**2, axis=1),  # Coloration basée sur la norme
                         cmap='viridis', 
                         alpha=0.6)
    plt.colorbar(scatter, label='Norme du vecteur latent')
    plt.title('Projection 2D de l\'espace latent (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
    plt.savefig('latent_space_rgb.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de: {device}")
    
    data_dir = "data"
    checkpoint_path = "best_model.pth"
    
    try:
        # Chargement du modèle
        model = MultiScaleAutoencoder().to(device)
        model, epoch, test_loss = load_checkpoint(model, checkpoint_path)
        print(f"Modèle chargé de l'époque {epoch} avec loss de test {test_loss:.6f}")
        
        # Chargement des données
        test_dataset = MultiScaleImageDataset(data_dir, 'test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Génération des visualisations
        print("Génération des reconstructions RGB...")
        visualize_random_samples(model, test_loader, device, num_samples=5)
        
        print("Génération de la visualisation de l'espace latent...")
        visualize_latent_space(model, test_loader, device)
        
        print("Visualisations terminées!")
        
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")
        raise

if __name__ == "__main__":
    main()