# BrainScan AI

**Détection et classification de tumeurs cérébrales à partir d'images IRM** à l'aide d'un réseau de
neurones convolutionnel (CNN).

Le modèle classe chaque IRM dans l'une des **4 catégories** :

| Classe | Description |
|---|---|
| `glioma` | Tumeur gliale |
| `meningioma` | Méningiome |
| `notumor` | Pas de tumeur |
| `pituitary` | Tumeur hypophysaire |

---

## Sommaire

- [Description](#description)
- [Jeu de données](#jeu-de-données)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Arborescence](#arborescence)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du carnet `02_cnn.ipynb`](#structure-du-carnet-02_cnnipynb)
- [Tests](#tests)
- [Gestion de projet (Jira)](#gestion-de-projet-jira)

---

## Description

Ce projet entraîne **from scratch** un CNN pour la classification d'IRM cérébrales (4 classes) et
fournit une **interface web Streamlit** (`app.py`) permettant de téléverser une IRM et d'obtenir la
prédiction assortie d'un score de confiance.

Tout le pipeline (chargement, prétraitement, modèle, entraînement, évaluation, inférence) est
implémenté **nativement dans `notebooks/02_cnn.ipynb`**, cellule par cellule, avec des explications
Markdown à chaque étape. Le dossier `src/` contient des squelettes modulaires destinés à une future
factorisation.

## Jeu de données

- Images IRM au format JPG réparties en 4 dossiers (un par classe) dans `Dataset/raw/`.
- **7 023 images** au total, taille d'entrée normalisée à **224 × 224 × 3** (RGB).

| Classe | Nombre d'images |
|---|---|
| glioma | 1621 |
| meningioma | 1645 |
| notumor | 2000 |
| pituitary | 1757 |

Le léger déséquilibre (ratio max/min ≈ 1.23) est corrigé par des **poids de classe** (`class_weight`)
calculés sur le jeu d'entraînement.

## Méthodologie

1. **Prétraitement** : filtrage des extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`), gestion des erreurs de
   lecture (`try-except`), redimensionnement `cv2.resize` 224×224, normalisation `[0, 1]` (division par
   `255.0`), encodage des labels (`LabelEncoder` + `to_categorical`).
2. **Split** stratifié : **68 % train / 12 % validation / 20 % test**.
3. **Modèle CNN** : blocs `Conv2D` (ReLU) → `MaxPooling2D` → `Dropout`, `GlobalAveragePooling2D`,
   `Dense` (ReLU), sortie `Dense(4, softmax)`. Optimiseur **Adam**, perte
   `categorical_crossentropy`, sauvegarde du meilleur modèle via **`ModelCheckpoint`**.
4. **Évaluation** : loss/accuracy sur le test, courbes d'apprentissage, matrice de confusion, rapport
   de classification (precision / recall / F1), analyse des prédictions correctes/incorrectes.
5. **Déploiement** : export du modèle en `.h5` + `class_mapping.json`, fonction `predict_image(path)`,
   interface Streamlit.

## Résultats

<!-- RESULTS -->

## Arborescence

```
brainscan/
├── app.py                     # Interface web Streamlit (upload d'IRM -> prédiction)
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
├── configs/
│   └── config.yaml            # Placeholder de configuration (YAML)
├── notebooks/
│   ├── 01_eda.ipynb           # Analyse exploratoire
│   └── 02_cnn.ipynb           # Pipeline complet : données -> CNN -> évaluation -> inférence
├── src/                       # Modules (squelettes) pour factorisation future
│   ├── config.py              # Chemins & constantes du projet
│   ├── data/                  # loader / preprocessing / splitting
│   ├── models/cnn.py
│   ├── training/              # train / callbacks
│   ├── evaluation/            # evaluate / plots
│   ├── inference/predictor.py
│   └── utils/                 # config / logging
├── tests/                     # Tests (squelettes pytest)
├── artifacts/                 # Sorties générées par le notebook
│   ├── figures/               # Graphiques (distribution, courbes, matrice de confusion…)
│   ├── metrics/               # test_metrics.json
│   └── models/                # brainscan_model.h5, brainscan_best_model.h5, class_mapping.json
└── Dataset/
    ├── raw/                   # IRM brutes (<classe>/<image>.jpg)
    └── processed/             # Tableaux .npy (optionnel)
```

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# (optionnel) binaire Graphviz pour plot_model — un schéma équivalent est généré sinon
#   Debian/Ubuntu : sudo apt-get install graphviz
#   macOS         : brew install graphviz
```

## Utilisation

```bash
# 1. Entraîner le modèle (génère le modèle et les artefacts)
jupyter lab notebooks/02_cnn.ipynb     # exécuter toutes les cellules

# 2. Lancer l'interface web de prédiction
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501` : téléversez une IRM (`.jpg`, `.jpeg`, `.png`,
`.bmp`) et obtenez la classe prédite, la confiance (%) et la distribution des probabilités.

## Structure du carnet `02_cnn.ipynb`

| Partie du cahier des charges | Section du notebook |
|---|---|
| 1. Chargement et prétraitement | cellules 1 → 13 |
| 2. Conception du modèle CNN | cellules 14 → 18 |
| 3. Entraînement et évaluation | cellules 19 → 27 |
| 4. Déploiement et utilisation | cellules 28 → 31 |
| 5. Documentation et bonnes pratiques | dernière cellule + ce README |

## Tests

```bash
pytest -q
```

Les fichiers `tests/*.py` sont des squelettes destinés aux tests unitaires du loader, du
prétraitement et du prédicteur.

## Gestion de projet (Jira)

La planification a été suivie dans Jira. Traduction du board en Epics et tickets :

### EPIC BSC-1 — Données & prétraitement
| Ticket | Description | Statut |
|---|---|---|
| BSC-11 | Import et exploration du dataset d'IRM | Terminé |
| BSC-12 | Filtrage des extensions + gestion des erreurs de lecture | Terminé |
| BSC-13 | Redimensionnement 224×224 et normalisation `[0, 1]` | Terminé |
| BSC-14 | Encodage des labels et split train/val/test stratifié | Terminé |
| BSC-15 | Analyse du déséquilibre et poids de classe | Terminé |

### EPIC BSC-2 — Modélisation
| Ticket | Description | Statut |
|---|---|---|
| BSC-21 | Architecture CNN (Conv2D / MaxPooling / Dropout / Dense) | Terminé |
| BSC-22 | Compilation (Adam, categorical_crossentropy) | Terminé |
| BSC-23 | Entraînement + ModelCheckpoint / EarlyStopping | Terminé |

### EPIC BSC-3 — Évaluation
| Ticket | Description | Statut |
|---|---|---|
| BSC-31 | Évaluation loss/accuracy sur le test | Terminé |
| BSC-32 | Courbes d'apprentissage, matrice de confusion, rapport de classification | Terminé |
| BSC-33 | Analyse des prédictions correctes/incorrectes | Terminé |

### EPIC BSC-4 — Déploiement
| Ticket | Description | Statut |
|---|---|---|
| BSC-41 | Fonction d'inférence `predict_image(path)` | Terminé |
| BSC-42 | Export du modèle (`.h5`) + mapping des classes | Terminé |
| BSC-43 | Application Streamlit (`app.py`) | Terminé |

### EPIC BSC-5 — Documentation & qualité
| Ticket | Description | Statut |
|---|---|---|
| BSC-51 | Notebook commenté + cellules Markdown | Terminé |
| BSC-52 | README (description, arborescence, commandes) | Terminé |
