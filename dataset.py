import os
import json
from datetime import datetime
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import shutil  # Pour déplacer les fichiers

class IGNMultiScaleDatasetBuilder:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "train")
        self.test_dir = os.path.join(base_dir, "test")
        self.wms_url = "https://data.geopf.fr/wms-r/wms"
        
    def get_bbox(self, lat, lon, size_meters):
        """Calcule la bbox autour d'un point central en mètres"""
        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * np.cos(np.radians(lat))
        
        half_size_deg_lat = (size_meters / 2) / meters_per_degree_lat
        half_size_deg_lon = (size_meters / 2) / meters_per_degree_lon
        
        return {
            'west': lon - half_size_deg_lon,
            'east': lon + half_size_deg_lon,
            'north': lat + half_size_deg_lat,
            'south': lat - half_size_deg_lat
        }
    
    def get_wms_url(self, bbox, size_px=64):
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': 'HR.ORTHOIMAGERY.ORTHOPHOTOS',
            'STYLES': '',
            'CRS': 'EPSG:4326',
            'BBOX': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
            'WIDTH': str(size_px),
            'HEIGHT': str(size_px),
            'FORMAT': 'image/jpeg'
        }
        
        url_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.wms_url}?{url_params}"

    def capture_location(self, lat, lon, name, subset="train"):
        """Capture les images aux 3 échelles pour une localisation donnée"""
        # Sélection du dossier approprié
        output_dir = self.train_dir if subset == "train" else self.test_dir
        
        scales = {
            'micro': 300,    # 300m × 300m
            'meso': 1000,    # 1000m × 1000m
            'macro': 4000    # 4km × 4km
        }
        
        location_dir = os.path.join(output_dir, name)
        os.makedirs(location_dir, exist_ok=True)
        
        metadata = {
            'latitude': lat,
            'longitude': lon,
            'capture_date': datetime.now().isoformat(),
            'scales': scales,
            'source': 'IGN-WMS',
            'subset': subset
        }
        
        with open(os.path.join(location_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        success = True
        for scale_name, scale_size in scales.items():
            image_path = os.path.join(location_dir, f'{scale_name}.png')
            bbox = self.get_bbox(lat, lon, scale_size)
            if not self.capture_ign_image(bbox, image_path):
                success = False
                break
                
        return success

    def capture_ign_image(self, bbox, output_path):
        """Capture une image IGN et la sauvegarde"""
        url = self.get_wms_url(bbox)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            img.save(output_path)
            print(f"Image sauvegardée : {output_path}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la capture de l'image: {e}")
            print(f"URL tentée: {url}")
            return False

def build_france_dataset(sample_size=100, test_split=0.2):
    """Construit un dataset sur la France métropolitaine avec split train/test"""
    bounds = {
        'min_lat': 41.3,  # Corse du Sud
        'max_lat': 51.1,  # Dunkerque
        'min_lon': -5.1,  # Pointe de Corsen
        'max_lon': 9.5    # Nice
    }
    
    # Calculer les tailles des ensembles
    test_size = int(sample_size * test_split)
    train_size = sample_size - test_size
    
    print(f"Création du dataset avec {train_size} exemples d'entraînement et {test_size} exemples de test")
    
    builder = IGNMultiScaleDatasetBuilder()
    os.makedirs(builder.train_dir, exist_ok=True)
    os.makedirs(builder.test_dir, exist_ok=True)
    
    # Générer tous les points
    np.random.seed(42)
    locations = []
    
    while len(locations) < sample_size:
        lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
        lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
        
        location = {
            'name': f'loc_{len(locations):04d}',
            'lat': lat,
            'lon': lon
        }
        locations.append(location)
    
    # Capturer les images pour l'ensemble d'entraînement
    print("\nCréation de l'ensemble d'entraînement...")
    for i, loc in enumerate(locations[:train_size]):
        print(f"\nCapturing train {loc['name']}... ({i+1}/{train_size})")
        print(f"Coordonnées : {loc['lat']:.4f}, {loc['lon']:.4f}")
        builder.capture_location(loc['lat'], loc['lon'], loc['name'], subset="train")
    
    # Capturer les images pour l'ensemble de test
    print("\nCréation de l'ensemble de test...")
    for i, loc in enumerate(locations[train_size:]):
        print(f"\nCapturing test {loc['name']}... ({i+1}/{test_size})")
        print(f"Coordonnées : {loc['lat']:.4f}, {loc['lon']:.4f}")
        builder.capture_location(loc['lat'], loc['lon'], loc['name'], subset="test")
    
    # Sauvegarder les statistiques du dataset
    stats = {
        'total_samples': sample_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'test_split': test_split,
        'creation_date': datetime.now().isoformat(),
        'bounds': bounds
    }
    
    with open(os.path.join(builder.base_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset créé avec succès!")
    print(f"Train: {train_size} exemples")
    print(f"Test: {test_size} exemples")

if __name__ == "__main__":
    # Créer un dataset avec 100 exemples, dont 20% pour le test
    build_france_dataset(sample_size=100000, test_split=0.01)