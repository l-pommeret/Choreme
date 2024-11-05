import os
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, upload_file
import json
from tqdm import tqdm
import torch

def upload_dataset_and_model_to_hf(
    data_dir="data",
    model_path="best_model_rgb.pth",
    hf_token="hf_votreTOKENici",  # Remplacez par votre token
    repo_id="votre-nom/nom-du-dataset"  # Format: username/dataset-name
):
    """
    Télécharge le dataset d'images aériennes et le modèle sur Hugging Face
    """
    # Initialisation de l'API Hugging Face
    api = HfApi(token=hf_token)

    # Création du repo si il n'existe pas
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True)
    except Exception as e:
        print(f"Le repo existe déjà ou erreur: {e}")

    def process_location(location_path, subset):
        """Traite un dossier de localisation et retourne ses données"""
        # Chargement des métadonnées
        with open(os.path.join(location_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Chemins des images
        image_paths = {
            'micro': os.path.join(location_path, 'micro.png'),
            'meso': os.path.join(location_path, 'meso.png'),
            'macro': os.path.join(location_path, 'macro.png')
        }
        
        # Vérification de la taille des images (doivent être 64x64)
        for img_path in image_paths.values():
            if not os.path.exists(img_path):
                return None
        
        return {
            'location_id': os.path.basename(location_path),
            'subset': subset,
            'latitude': metadata['latitude'],
            'longitude': metadata['longitude'],
            'micro_image': image_paths['micro'],
            'meso_image': image_paths['meso'],
            'macro_image': image_paths['macro'],
        }

    # Collecte des données
    dataset_dict = {'train': [], 'test': []}
    
    # Traitement des ensembles train et test
    for subset in ['train', 'test']:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            continue
            
        for location in tqdm(os.listdir(subset_dir), desc=f"Processing {subset}"):
            location_path = os.path.join(subset_dir, location)
            if os.path.isdir(location_path):
                data = process_location(location_path, subset)
                if data:
                    dataset_dict[subset].append(data)

    # Création des features
    features = Features({
        'location_id': Value('string'),
        'subset': Value('string'),
        'latitude': Value('float64'),
        'longitude': Value('float64'),
        'micro_image': Image(),
        'meso_image': Image(),
        'macro_image': Image(),
    })

    # Upload des datasets
    for subset, data in dataset_dict.items():
        if data:
            ds = Dataset.from_list(data, features=features)
            ds.push_to_hub(
                repo_id,
                split=subset,
                token=hf_token,
                private=True
            )

    # Upload du modèle entraîné si disponible
    if os.path.exists(model_path):
        print("\nTéléchargement du modèle entraîné...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Sauvegarde des hyperparamètres du modèle
        model_config = {
            'scale_latent_dim': 64,  # valeurs par défaut de votre modèle
            'final_latent_dim': 128,
            'image_size': 64,
            'channels': 3,
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss'],
            'test_loss': checkpoint['test_loss']
        }
        
        # Sauvegarde de la configuration
        config_path = 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
            
        # Upload de la configuration et du modèle
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="model_config.json",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model_rgb.pth",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        os.remove(config_path)  # Nettoyage

    # Upload des statistiques du dataset
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.exists(stats_path):
        api.upload_file(
            path_or_fileobj=stats_path,
            path_in_repo="dataset_stats.json",
            repo_id=repo_id,
            repo_type="dataset"
        )

    print(f"\nDataset et modèle téléchargés avec succès sur {repo_id}")
    print("Vous pouvez maintenant accéder à votre dataset sur Hugging Face!")

if __name__ == "__main__":
    upload_dataset_and_model_to_hf(
        data_dir="data",
        model_path="best_model_rgb.pth",  # Chemin vers votre modèle entraîné
        hf_token="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",  # Remplacez par votre token
        repo_id="Zual/choreme" # Remplacez par votre nom d'utilisateur et le nom souhaité pour le dataset
    )