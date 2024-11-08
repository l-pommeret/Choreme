import os
import json
from datasets import load_dataset
from PIL import Image
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import shutil

def download_dataset_from_hf(
    repo_id="Zual/choreme",
    output_dir="data",
    hf_token="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"
):
    """
    Télécharge le dataset depuis Hugging Face et recrée la structure de dossiers
    """
    print(f"Téléchargement du dataset depuis {repo_id}...")

    # Création des dossiers
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Chargement du dataset avec la nouvelle méthode d'authentification
    try:
        dataset = load_dataset(
            repo_id,
            token=hf_token  # Nouveau paramètre au lieu de use_auth_token
        )
    except Exception as e:
        print(f"Erreur lors du chargement initial, tentative alternative... ({e})")
        # Tentative alternative avec configuration manuelle
        from huggingface_hub import HfFolder
        HfFolder.save_token(hf_token)
        dataset = load_dataset(repo_id)

    def save_location(example, subset_dir):
        """Sauvegarde une location avec ses images et métadonnées"""
        location_dir = os.path.join(subset_dir, example['location_id'])
        os.makedirs(location_dir, exist_ok=True)

        # Sauvegarde des images
        for scale in ['micro', 'meso', 'macro']:
            img = example[f'{scale}_image']
            if isinstance(img, dict):  # Si l'image est dans le format du dataset HF
                img = Image.fromarray(img['array'])
            img.save(os.path.join(location_dir, f'{scale}.png'))

        # Création des métadonnées
        metadata = {
            'latitude': float(example['latitude']),
            'longitude': float(example['longitude']),
            'subset': example['subset'],
            'scales': {
                'micro': 300,
                'meso': 1000,
                'macro': 4000
            }
        }

        # Sauvegarde des métadonnées
        with open(os.path.join(location_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    # Traitement de l'ensemble d'entraînement
    if 'train' in dataset:
        print("\nTraitement de l'ensemble d'entraînement...")
        for example in tqdm(dataset['train']):
            save_location(example, train_dir)

    # Traitement de l'ensemble de test
    if 'test' in dataset:
        print("\nTraitement de l'ensemble de test...")
        for example in tqdm(dataset['test']):
            save_location(example, test_dir)

    # Téléchargement des statistiques du dataset
    try:
        stats_path = hf_hub_download(
            repo_id=repo_id,
            filename="dataset_stats.json",
            repo_type="dataset",
            token=hf_token
        )
        shutil.copy(stats_path, os.path.join(output_dir, 'dataset_stats.json'))
    except Exception as e:
        print(f"Note: Impossible de télécharger dataset_stats.json: {e}")

    # Téléchargement du modèle si disponible
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="best_model_rgb.pth",
            repo_type="dataset",
            token=hf_token
        )
        shutil.copy(model_path, "best_model_rgb.pth")
        print("\nModèle téléchargé avec succès !")
    except Exception as e:
        print(f"Note: Impossible de télécharger le modèle: {e}")

    print(f"\nDataset téléchargé avec succès dans {output_dir}!")
    print("Structure créée:")
    print(f"- {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   └── loc_xxxx/")
    print(f"  │       ├── micro.png")
    print(f"  │       ├── meso.png")
    print(f"  │       ├── macro.png")
    print(f"  │       └── metadata.json")
    print(f"  └── test/")
    print(f"      └── [même structure]")

if __name__ == "__main__":
    download_dataset_from_hf(
        repo_id="Zual/choreme",  # Votre repo
        output_dir="data",  # Dossier de sortie
        hf_token="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"  # Votre token
    )