# Introduction à NumPy

## Qu'est-ce que NumPy ?
NumPy (Numerical Python) est la bibliothèque fondamentale pour le calcul scientifique en Python. Elle fournit des structures de données puissantes (principalement les tableaux `ndarray`) et des fonctions optimisées en C pour des opérations mathématiques et logiques sur des tableaux multidimensionnels.

Pourquoi NumPy est important :
- Performance : les opérations vectorisées sont beaucoup plus rapides que les boucles Python pures.
- Interopérabilité : base pour pandas, scikit-learn, matplotlib et la plupart des outils de data science / machine learning.
- API compacte : manipulation d'indices, diffusion (broadcasting), formes (reshape) et opérations universelles (ufuncs).

## Comment utiliser ce dossier
- Notebook d'apprentissage : `NUMPY/learning_numpy.ipynb` (exemples étape par étape).
- Ce fichier donne une vue rapide et des exemples prêts à exécuter pour démarrer.

## Premiers pas — Exemples de manipulation de données
Les exemples ci-dessous sont prêts à être copiés dans un script Python ou une cellule notebook.

1) Import et création de tableaux

```python
import numpy as np

# création d'un tableau 1D
a = np.array([1, 2, 3, 4])

# création d'un tableau 2D
b = np.array([[1, 2], [3, 4]])

# création d'array avec des fonctions utilitaires
zeros = np.zeros((3, 4))
ones = np.ones(5)
rand = np.random.rand(3, 3)  # valeurs uniformes [0,1)

print(a, b.shape)
```

2) Opérations vectorisées et broadcasting

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

# addition élément par élément (vectorisé)
print(x + y)

# broadcasting: ajout d'un scalaire
print(x + 5)
```

3) Indexation, slicing et reshape

```python
arr = np.arange(12)            # [0..11]
arr2 = arr.reshape(3, 4)       # forme 3x4

# slicing
print(arr2[1, :])              # deuxième ligne
print(arr2[:, 2])              # troisième colonne

# vues vs copies : attention aux modifications in-place
view = arr2[:, :2]
view[0, 0] = 999
# arr2 est impacté car view est une vue
```

4) Statistiques et agrégations

```python
data = np.random.randn(1000)
print(data.mean(), data.std(), data.sum())

# opérations le long d'un axe
mat = np.random.rand(4, 5)
print(mat.sum(axis=0))  # somme par colonne
```

5) Lecture/écriture rapide d'array

```python
np.save('mon_array.npy', arr2)
loaded = np.load('mon_array.npy')
```

6) Interopérabilité avec pandas

```python
import pandas as pd

series = pd.Series(np.random.rand(5), name='valeurs')
df = pd.DataFrame(np.random.randn(4,3), columns=['a','b','c'])
# pandas repose sur NumPy : conversions aisées
arr_from_df = df.values  # numpy ndarray
```

## Commandes pratiques (environnement local)
- Activer l'environnement virtuel Python (si vous utilisez l'env fourni dans le dépôt) :

```bash
source env/bin/activate
python -V
pip install -U numpy
```

- Ouvrir le notebook d'apprentissage :

```bash
# depuis la racine du dépôt
jupyter notebook NUMPY/learning_numpy.ipynb
```

### Installation — Windows et Linux

Voici des instructions concrètes et reproductibles pour installer NumPy selon votre plateforme et votre préférence d'environnement.

1) Recommandation générale (virtualenv / venv — multiplateforme)

```bash
# créer et activer un environnement virtuel (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# sur Windows PowerShell
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install numpy
```

2) Installation rapide avec pip (sans venv — utilisateur)

```bash
# Linux / macOS
pip install --user numpy

# Windows (PowerShell)
pip install --user numpy
```

3) Utilisateurs conda (optionnel)

```bash
# créer un env et installer numpy
conda create -n ds-env python=3.12 numpy -y
conda activate ds-env
```

4) Notes spécifiques pour Linux

- Si pip n'est pas installé :

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv    # Debian/Ubuntu
```

- Pour de meilleures performances numériques (OpenBLAS/MKL), installez la version fournie par la distribution (ou utilisez conda qui gère les BLAS correctement) :

```bash
pip install numpy            # version standard (utilise les wheels précompilés)
# ou via conda pour MKL/optimisations:
conda install numpy
```

5) Vérifier l'installation

```bash
python -c "import numpy as np; print('numpy', np.__version__, 'blas:', np.__config__.get_info('blas') or 'unknown')"
```

6) Si vous utilisez l'environnement `env/` fourni dans le dépôt

```bash
source env/bin/activate
pip install -U numpy
```

Ces instructions couvrent les scénarios les plus courants (venv, pip --user, conda) sur Windows et Linux. Dites-moi si vous voulez que j'ajoute :
- instructions PowerShell plus détaillées, ou
- une vérification automatisée (`NUMPY/check_env.py`) qui teste les opérations de base.


## Conventions et bonnes pratiques observées dans ce dépôt
- Le dossier `NUMPY/` contient des notebooks pédagogiques orientés pas-à-pas.
- D'autres dossiers `PANDAS/`, `SEABO             RN/`, `MATPLOTLIB/` contiennent notebooks thématiques — attendez-vous à interconnectivité (ex. conversion DataFrame <-> ndarray).
- Utiliser l'environnement `env/` fourni pour reproduire les versions de packages présentes.

## Prochaines étapes suggérées
1. Ouvrir `NUMPY/learning_numpy.ipynb` et exécuter cellule par cellule.
2. Essayer les exemples ci-dessus dans une cellule de notebook pour sentir la différence entre vues et copies.
3. Expérimenter l'intégration avec `PANDAS/learning_pandas.ipynb` (conversion DataFrame ↔ ndarray).

Si vous voulez, je peux :
- Ajouter des exercices guidés avec solutions dans le notebook.
- Fournir un petit script de tests pour valider l'environnement NumPy de `env/`.

---
Fichier créé automatiquement : si vous souhaitez plus de détails (e.g., sections sur broadcasting avancé, performance ou numba), dites-le et j'ajoute une section ciblée.
