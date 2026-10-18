import os

# ==========================================
# CHEMINS DU PROJET (PATHS)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers des données
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")

# Dossiers des artefacts (modèles, métriques, figures)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

# Fichier de mapping des classes
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "brainscan_best_model.h5")

# ==========================================
# PARAMÈTRES DES IMAGES ET DU PRÉTRAITEMENT
# ==========================================
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3  # Images en RGB
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# Ratios de division des données
TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

# ==========================================
# HYPERPARAMÈTRES DU MODÈLE CNN
# ==========================================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
LOSS_FUNCTION = "sparse_categorical_crossentropy" # ou "categorical_crossentropy" selon l'encodage
METRICS = ["accuracy"]

# Classes cibles du projet BrainScan AI
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)