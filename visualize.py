import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

from autoencoder import MultiScaleCNNAutoencoder, MultiScaleImageDataset

def load_trained_model(model_path='best_model.pth', device='cuda'):
    """Charge le modèle entraîné"""
    model = MultiScaleCNNAutoencoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_random_sample(model, dataset, device='cuda'):
    """Visualise un échantillon aléatoire et sa reconstruction"""
    # Sélectionner un index aléatoire
    idx = random.randint(0, len(dataset)-1)
    micro, meso, macro = dataset[idx]

    # Préparer les données pour le modèle
    micro = micro.unsqueeze(0).to(device)
    meso = meso.unsqueeze(0).to(device)
    macro = macro.unsqueeze(0).to(device)

    # Générer les reconstructions
    with torch.no_grad():
        micro_rec, meso_rec, macro_rec, _ = model(micro, meso, macro)

    # Convertir les tensors en images numpy
    def to_image(tensor):
        return tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Images originales
    micro_img = to_image(micro)
    meso_img = to_image(meso)
    macro_img = to_image(macro)

    # Images reconstruites
    micro_rec_img = to_image(micro_rec)
    meso_rec_img = to_image(meso_rec)
    macro_rec_img = to_image(macro_rec)

    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle('Comparaison Original vs Reconstruction', fontsize=16)

    # Première ligne : images originales
    axes[0, 0].imshow(micro_img)
    axes[0, 0].set_title('Micro Original')
    axes[0, 1].imshow(meso_img)
    axes[0, 1].set_title('Meso Original')
    axes[0, 2].imshow(macro_img)
    axes[0, 2].set_title('Macro Original')

    # Deuxième ligne : reconstructions
    axes[1, 0].imshow(micro_rec_img)
    axes[1, 0].set_title('Micro Reconstruit')
    axes[1, 1].imshow(meso_rec_img)
    axes[1, 1].set_title('Meso Reconstruit')
    axes[1, 2].imshow(macro_rec_img)
    axes[1, 2].set_title('Macro Reconstruit')

    # Ajuster la mise en page
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()

    # Sauvegarder et afficher
    plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Retourner les images pour d'autres utilisations potentielles
    return {
        'original': (micro_img, meso_img, macro_img),
        'reconstructed': (micro_rec_img, meso_rec_img, macro_rec_img)
    }

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"  # Ajuster selon votre configuration

    # Charger le dataset et le modèle
    dataset = MultiScaleImageDataset(data_dir, subset='test')
    model = load_trained_model(device=device)

    # Visualiser un échantillon aléatoire
    images = visualize_random_sample(model, dataset, device)