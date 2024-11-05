import os
import json
from datetime import datetime
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import shutil

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
        """Génère l'URL WMS pour la requête d'image"""
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

    def check_white_percentage(self, image_path, threshold=0.1):
        """
        Vérifie si l'image contient plus qu'un certain pourcentage de pixels blancs
        Retourne True si l'image est valide (moins de blancs que le threshold)
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Convertir en RGB si l'image est en RGBA
            if img_array.shape[-1] == 4:
                img = img.convert('RGB')
                img_array = np.array(img)
            
            # Compter les pixels blancs (ou presque blancs)
            white_pixels = np.sum(np.all(img_array > 250, axis=-1))
            total_pixels = img_array.shape[0] * img_array.shape[1]
            white_ratio = white_pixels / total_pixels
            
            return white_ratio <= threshold
            
        except Exception as e:
            print(f"Erreur lors de la vérification des pixels blancs: {e}")
            return False

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

    def cleanup_location(self, location_dir):
        """Nettoie le dossier de location en cas d'échec"""
        try:
            if os.path.exists(location_dir):
                shutil.rmtree(location_dir)
                print(f"Dossier nettoyé : {location_dir}")
        except Exception as e:
            print(f"Erreur lors du nettoyage du dossier {location_dir}: {e}")

    def capture_location(self, lat, lon, name, subset="train"):
        """Capture les images aux 3 échelles pour une localisation donnée"""
        output_dir = self.train_dir if subset == "train" else self.test_dir
        
        scales = {
            'micro': 300,    # 300m × 300m
            'meso': 1000,    # 1000m × 1000m
            'macro': 4000    # 4km × 4km
        }
        
        location_dir = os.path.join(output_dir, name)
        os.makedirs(location_dir, exist_ok=True)
        
        # Commencer par capturer l'image micro
        bbox = self.get_bbox(lat, lon, scales['micro'])
        micro_path = os.path.join(location_dir, 'micro.png')
        
        if not self.capture_ign_image(bbox, micro_path):
            print("Échec de la capture de l'image micro")
            self.cleanup_location(location_dir)
            return False
            
        # Vérifier le pourcentage de blanc dans l'image micro
        if not self.check_white_percentage(micro_path):
            print(f"Image micro rejetée car trop de blanc: {micro_path}")
            self.cleanup_location(location_dir)
            return False
            
        # Si l'image micro est valide, sauvegarder les métadonnées
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
        
        # Capturer les autres échelles
        for scale_name, scale_size in {k: v for k, v in scales.items() if k != 'micro'}.items():
            image_path = os.path.join(location_dir, f'{scale_name}.png')
            bbox = self.get_bbox(lat, lon, scale_size)
            if not self.capture_ign_image(bbox, image_path):
                print(f"Échec de la capture de l'image {scale_name}")
                self.cleanup_location(location_dir)
                return False
                
        return True

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
    successful_train = 0
    successful_test = 0
    location_index = 0
    
    while successful_train < train_size:
        lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
        lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
        
        print(f"\nTentative train {successful_train + 1}/{train_size}")
        print(f"Coordonnées : {lat:.4f}, {lon:.4f}")
        
        if builder.capture_location(lat, lon, f'loc_{location_index:04d}', subset="train"):
            successful_train += 1
        location_index += 1
    
    while successful_test < test_size:
        lat = np.random.uniform(bounds['min_lat'], bounds['max_lat'])
        lon = np.random.uniform(bounds['min_lon'], bounds['max_lon'])
        
        print(f"\nTentative test {successful_test + 1}/{test_size}")
        print(f"Coordonnées : {lat:.4f}, {lon:.4f}")
        
        if builder.capture_location(lat, lon, f'loc_{location_index:04d}', subset="test"):
            successful_test += 1
        location_index += 1
    
    # Sauvegarder les statistiques du dataset
    stats = {
        'total_successful_samples': successful_train + successful_test,
        'total_attempts': location_index,
        'train_samples': successful_train,
        'test_samples': successful_test,
        'test_split': test_split,
        'creation_date': datetime.now().isoformat(),
        'bounds': bounds,
        'success_rate': (successful_train + successful_test) / location_index
    }
    
    with open(os.path.join(builder.base_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset créé avec succès!")
    print(f"Train: {successful_train} exemples")
    print(f"Test: {successful_test} exemples")
    print(f"Taux de succès: {stats['success_rate']:.2%}")

if __name__ == "__main__":
    build_france_dataset(sample_size=100000, test_split=0.01)