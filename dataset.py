import os
import json
from datetime import datetime
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import math

class IGNMultiScaleDatasetBuilder:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.wmts_url = "https://data.geopf.fr/wmts"
        
    def deg2tile(self, lat_deg, lon_deg, zoom):
        """Convertit lat/lon en coordonnées de tuile"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def get_wmts_url(self, lat, lon, zoom):
        """Génère l'URL WMTS pour l'IGN"""
        x, y = self.deg2tile(lat, lon, zoom)
        
        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetTile',
            'VERSION': '1.0.0',
            'LAYER': 'ORTHOIMAGERY.ORTHOPHOTOS',
            'STYLE': 'normal',
            'FORMAT': 'image/jpeg',
            'TILEMATRIXSET': 'PM',
            'TILEMATRIX': str(zoom),
            'TILEROW': str(y),
            'TILECOL': str(x)
        }
        
        # Construire l'URL avec les paramètres
        url_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.wmts_url}?{url_params}"

    def get_appropriate_zoom(self, size_meters):
        """Détermine le niveau de zoom approprié pour une taille donnée"""
        # Approximation basique :
        # zoom 18 ≈ 50m
        # zoom 16 ≈ 200m
        # zoom 14 ≈ 800m
        # zoom 12 ≈ 3200m
        if size_meters <= 100:
            return 18
        elif size_meters <= 500:
            return 16
        else:
            return 14

    def capture_location(self, lat, lon, name):
        """Capture les images aux 3 échelles pour une localisation donnée"""
        # Définition des échelles
        scales = {
            'micro': {'size': 100, 'zoom': 18},   # 100m × 100m
            'meso': {'size': 500, 'zoom': 16},    # 500m × 500m
            'macro': {'size': 2000, 'zoom': 14}   # 2km × 2km
        }
        
        # Créer le dossier pour ce lieu
        location_dir = os.path.join(self.output_dir, name)
        os.makedirs(location_dir, exist_ok=True)
        
        # Métadonnées
        metadata = {
            'latitude': lat,
            'longitude': lon,
            'capture_date': datetime.now().isoformat(),
            'scales': {k: v['size'] for k, v in scales.items()},
            'source': 'IGN-WMTS'
        }
        
        # Sauvegarder les métadonnées
        with open(os.path.join(location_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Capturer les images à chaque échelle
        for scale_name, scale_info in scales.items():
            image_path = os.path.join(location_dir, f'{scale_name}.png')
            self.capture_ign_image(lat, lon, scale_info['zoom'], image_path)

    def capture_ign_image(self, lat, lon, zoom, output_path):
        """Capture une image IGN et la sauvegarde"""
        url = self.get_wmts_url(lat, lon, zoom)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convertir la réponse en image
            img = Image.open(BytesIO(response.content))
            
            # Redimensionner en 64x64
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Sauvegarder
            img.save(output_path)
            print(f"Image sauvegardée : {output_path}")
            
        except Exception as e:
            print(f"Erreur lors de la capture de l'image: {e}")
            print(f"URL tentée: {url}")
            return None

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