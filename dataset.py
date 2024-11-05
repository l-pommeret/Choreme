import os
import json
from datetime import datetime
import requests
from PIL import Image
import numpy as np
from io import BytesIO

class IGNMultiScaleDatasetBuilder:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.wms_url = "https://data.geopf.fr/wms-r/wms"
        
    def get_bbox(self, lat, lon, size_meters):
        """Calcule la bbox autour d'un point central en mètres"""
        # Conversion approximative degrés -> mètres (à une latitude moyenne en France)
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
        """Génère l'URL WMS pour l'IGN"""
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetMap',
            'LAYERS': 'HR.ORTHOIMAGERY.ORTHOPHOTOS',  # Couche orthophoto haute résolution
            'STYLES': '',
            'CRS': 'EPSG:4326',
            'BBOX': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
            'WIDTH': str(size_px),
            'HEIGHT': str(size_px),
            'FORMAT': 'image/jpeg'
        }
        
        # Construire l'URL avec les paramètres
        url_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.wms_url}?{url_params}"

    def capture_location(self, lat, lon, name):
        """Capture les images aux 3 échelles pour une localisation donnée"""
        # Définition des échelles
        scales = {
            'micro': 100,   # 100m × 100m
            'meso': 500,    # 500m × 500m
            'macro': 2000   # 2km × 2km
        }
        
        # Créer le dossier pour ce lieu
        location_dir = os.path.join(self.output_dir, name)
        os.makedirs(location_dir, exist_ok=True)
        
        # Métadonnées
        metadata = {
            'latitude': lat,
            'longitude': lon,
            'capture_date': datetime.now().isoformat(),
            'scales': scales,
            'source': 'IGN-WMS'
        }
        
        # Sauvegarder les métadonnées
        with open(os.path.join(location_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Capturer les images à chaque échelle
        for scale_name, scale_size in scales.items():
            image_path = os.path.join(location_dir, f'{scale_name}.png')
            bbox = self.get_bbox(lat, lon, scale_size)
            self.capture_ign_image(bbox, image_path)

    def capture_ign_image(self, bbox, output_path):
        """Capture une image IGN et la sauvegarde"""
        url = self.get_wms_url(bbox)
        
        try:
            response = requests.get(url, timeout=30)  # Timeout augmenté car WMS plus lent
            response.raise_for_status()
            
            # Convertir la réponse en image
            img = Image.open(BytesIO(response.content))
            
            # Déjà en 64x64 grâce aux paramètres WMS
            img.save(output_path)
            print(f"Image sauvegardée : {output_path}")
            return True
            
        except Exception as e:
            print(f"Erreur lors de la capture de l'image: {e}")
            print(f"URL tentée: {url}")
            return False

def build_france_dataset(sample_size=10):
    """Construit un dataset sur la France métropolitaine"""
    # Limites approximatives de la France métropolitaine
    bounds = {
        'min_lat': 41.3,  # Corse du Sud
        'max_lat': 51.1,  # Dunkerque
        'min_lon': -5.1,  # Pointe de Corsen
        'max_lon': 9.5    # Nice
    }
    
    builder = IGNMultiScaleDatasetBuilder()
    
    # Générer des points aléatoires dans les limites de la France
    np.random.seed(42)  # Pour la reproductibilité
    
    # Liste pour stocker les points générés
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
    
    # Capturer les images pour chaque location
    for loc in locations:
        print(f"\nCapturing {loc['name']}...")
        print(f"Coordonnées : {loc['lat']:.4f}, {loc['lon']:.4f}")
        builder.capture_location(loc['lat'], loc['lon'], loc['name'])

if __name__ == "__main__":
    # Créer un petit dataset de test
    build_france_dataset(sample_size=10)