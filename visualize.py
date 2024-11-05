import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import MultiScaleCNNAutoencoder, MultiScaleImageDataset  # Importer depuis votre fichier model.py
import os

def plot_sample(original, reconstructed, titles, save_path):
    """Plot une comparaison côte à côte"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    ax1.imshow(original)
    ax1.set_title(titles[0])
    ax1.axis('off')
    
    # Reconstruction
    ax2.imshow(reconstructed)
    ax2.set_title(titles[1])
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_results():
    # Configuration
    data_dir = "data"
    save_dir = "results"
    model_path = "best_model.pth"
    
    # Créer le dossier de résultats s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleCNNAutoencoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger quelques données de test
    dataset = MultiScaleImageDataset(data_dir, subset='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Sélectionner 3 échantillons
    with torch.no_grad():
        for i, (micro, meso, macro) in enumerate(dataloader):
            if i >= 3:  # Visualiser seulement 3 échantillons
                break
                
            # Déplacer les données sur le GPU
            micro = micro.to(device)
            meso = meso.to(device)
            macro = macro.to(device)
            
            # Obtenir les reconstructions
            micro_recon, meso_recon, macro_recon, _ = model(micro, meso, macro)
            
            # Convertir en numpy et ajuster le format
            micro = micro[0].cpu().numpy().transpose(1, 2, 0)
            meso = meso[0].cpu().numpy().transpose(1, 2, 0)
            macro = macro[0].cpu().numpy().transpose(1, 2, 0)
            
            micro_recon = micro_recon[0].cpu().numpy().transpose(1, 2, 0)
            meso_recon = meso_recon[0].cpu().numpy().transpose(1, 2, 0)
            macro_recon = macro_recon[0].cpu().numpy().transpose(1, 2, 0)
            
            # Sauvegarder les comparaisons
            plot_sample(micro, micro_recon, 
                       ['Original Micro', 'Reconstructed Micro'],
                       f'{save_dir}/sample_{i+1}_micro.png')
            
            plot_sample(meso, meso_recon,
                       ['Original Meso', 'Reconstructed Meso'],
                       f'{save_dir}/sample_{i+1}_meso.png')
            
            plot_sample(macro, macro_recon,
                       ['Original Macro', 'Reconstructed Macro'],
                       f'{save_dir}/sample_{i+1}_macro.png')
            
            # Créer une visualisation combinée
            fig, axes = plt.subplots(3, 2, figsize=(12, 18))
            plt.suptitle(f'Sample {i+1} - All Scales Comparison', size=16)
            
            # Première rangée : Micro
            axes[0,0].imshow(micro)
            axes[0,0].set_title('Original Micro')
            axes[0,0].axis('off')
            axes[0,1].imshow(micro_recon)
            axes[0,1].set_title('Reconstructed Micro')
            axes[0,1].axis('off')
            
            # Deuxième rangée : Meso
            axes[1,0].imshow(meso)
            axes[1,0].set_title('Original Meso')
            axes[1,0].axis('off')
            axes[1,1].imshow(meso_recon)
            axes[1,1].set_title('Reconstructed Meso')
            axes[1,1].axis('off')
            
            # Troisième rangée : Macro
            axes[2,0].imshow(macro)
            axes[2,0].set_title('Original Macro')
            axes[2,0].axis('off')
            axes[2,1].imshow(macro_recon)
            axes[2,1].set_title('Reconstructed Macro')
            axes[2,1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{i+1}_all_scales.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculer et afficher les métriques
            def compute_metrics(orig, recon):
                mse = np.mean((orig - recon) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                return mse, psnr
            
            micro_mse, micro_psnr = compute_metrics(micro, micro_recon)
            meso_mse, meso_psnr = compute_metrics(meso, meso_recon)
            macro_mse, macro_psnr = compute_metrics(macro, macro_recon)
            
            print(f"\nSample {i+1} Metrics:")
            print(f"Micro - MSE: {micro_mse:.6f}, PSNR: {micro_psnr:.2f} dB")
            print(f"Meso  - MSE: {meso_mse:.6f}, PSNR: {meso_psnr:.2f} dB")
            print(f"Macro - MSE: {macro_mse:.6f}, PSNR: {macro_psnr:.2f} dB")

if __name__ == "__main__":
    print("Generating visualizations...")
    visualize_results()
    print("\nDone! Check the 'results' folder for the visualizations.")