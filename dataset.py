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
        self.wmts_url = "https://wxs.ign.fr/ortho/geoportail/wmts"
        self.api_key = "essentiels"  # clé gratuite pour les données essentielles
        
    def get_bbox_from_center(self, lat, lon, size_meters):
        """Calcule la bbox autour d'un point central"""
        # Approximation rapide (à affiner selon la latitude)
        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * np.cos(np.radians(lat))
        
        half_size_lat = (size_meters / 2) / meters_per_degree_lat
        half_size_lon = (size_meters / 2) / meters_per_degree_lon
        
        return {
            'west': lon - half_size_lon,
            'east': lon + half_size_lon,
            'north': lat + half_size_lat,
            'south': lat - half_size_lat
        }

    def get_wmts_url(self, bbox, size_px=64):
        """Génère l'URL WMTS pour l'IGN"""
        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetTile',
            'VERSION': '1.0.0',
            'LAYER': 'ORTHOIMAGERY.ORTHOPHOTOS',
            'STYLE': 'normal',
            'FORMAT': 'image/jpeg',
            'TILEMATRIXSET': 'PM',
            'TILEMATRIX': '16',
            'TILEROW': '23294',
            'TILECOL': '33536',
            'apikey': self.api_key
        }
        
        # Construire l'URL avec les paramètres
        url_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.wmts_url}?{url_params}"

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
            'source': 'IGN-WMTS'
        }
        
        # Sauvegarder les métadonnées
        with open(os.path.join(location_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Capturer les images à chaque échelle
        for scale_name, scale_size in scales.items():
            image_path = os.path.join(location_dir, f'{scale_name}.png')
            bbox = self.get_bbox_from_center(lat, lon, scale_size)
            self.capture_ign_image(bbox, image_path)

    def capture_ign_image(self, bbox, output_path):
        """Capture une image IGN et la sauvegarde"""
        url = self.get_wmts_url(bbox)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Convertir la réponse en image
            img = Image.open(BytesIO(response.content))
            
            # Redimensionner en 64x64
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Sauvegarder
            img.save(output_path)
            
        except Exception as e:
            print(f"Erreur lors de la capture de l'image: {e}")
            return None

def build_france_dataset(sample_size=1000):
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
        
        # TODO: Ajouter une vérification si le point est bien en France
        # On pourrait utiliser un shapefile de la France pour ça
        
        location = {
            'name': f'loc_{len(locations):04d}',
            'lat': lat,
            'lon': lon
        }
        locations.append(location)
    
    # Capturer les images pour chaque location
    for loc in locations:
        print(f"Capturing {loc['name']}...")
        builder.capture_location(loc['lat'], loc['lon'], loc['name'])

if __name__ == "__main__":
    # Créer un petit dataset de test
    build_france_dataset(sample_size=10)