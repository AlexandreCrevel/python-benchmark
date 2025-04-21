# Projet de Benchmark Python Simple

Ce projet fournit un ensemble de scripts Python pour évaluer et comparer les performances de différentes machines (par exemple, un PC Windows et un MacBook) sur diverses tâches de calcul.

## Prérequis

- **Python 3**: Assurez-vous que Python 3 (version 3.7 ou supérieure recommandée) est installé sur les deux machines.
- **pip**: Le gestionnaire de paquets Python, généralement inclus avec Python.

## Installation

1.  **Clonez ou copiez ce projet** sur les deux machines que vous souhaitez comparer. Placez-le dans un répertoire de votre choix.
2.  **Ouvrez un terminal ou une invite de commande** dans le dossier `python-benchmark`.
3.  **(Recommandé)** Créez et activez un environnement virtuel :

    ```bash
    # Sur macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Sur Windows (cmd)
    python -m venv venv
    venv\Scripts\activate

    # Sur Windows (PowerShell)
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

4.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

## Exécution des Benchmarks

1.  **Assurez-vous qu'aucune autre application gourmande en ressources** ne tourne en arrière-plan sur la machine pour obtenir des résultats plus cohérents.
2.  **Exécutez le script principal** depuis le terminal (assurez-vous que votre environnement virtuel est activé si vous en utilisez un) :
    ```bash
    python benchmark.py
    ```
3.  Le script exécutera séquentiellement chaque benchmark et affichera le temps pris pour chacun. Un résumé final sera également affiché.
4.  **Répétez l'opération** sur l'autre machine en utilisant **exactement le même code et les mêmes versions de bibliothèques** (grâce à `requirements.txt` et à l'environnement virtuel).

## Les Benchmarks

- **Fibonacci (CPU)**: Teste la performance brute du CPU sur des appels de fonctions récursives (single-thread).
- **Nombres Premiers (CPU)**: Teste la performance du CPU sur des calculs et boucles intensifs (single-thread).
- **Multiplication de Matrices (CPU/SIMD)**: Utilise NumPy pour tester les capacités de calcul numérique, souvent optimisées via SIMD et des bibliothèques C/Fortran sous-jacentes. Peut utiliser plusieurs cœurs selon l'installation NumPy/BLAS.
- **Opérations sur Listes (Mémoire/CPU)**: Mesure le temps nécessaire pour allouer une grande quantité de mémoire pour une liste Python et effectuer une opération dessus (tri).
- **I/O Disque - Gros Fichier**: Teste la vitesse de lecture et d'écriture séquentielle sur le disque dur/SSD.
- **I/O Disque - Petits Fichiers**: Teste la performance du système de fichiers pour créer, écrire et lire de nombreux petits fichiers, ce qui est courant lors des builds de projets ou de la gestion de caches.
- **Hashing (CPU/Mémoire)**: Mesure la vitesse de calcul d'un hash SHA-256 sur une quantité de données en mémoire.
- **Compression/Décompression (CPU)**: Teste la performance du CPU sur des algorithmes de compression/décompression (zlib).

## Interprétation des Résultats

- Comparez les temps (en secondes) obtenus pour chaque benchmark entre les deux machines.
- **Un temps plus court indique une meilleure performance** pour ce test spécifique.
- Notez les différences relatives. Une machine peut exceller en CPU pur mais être plus lente en I/O disque, par exemple.
- Pour des résultats plus fiables, vous pouvez exécuter le script plusieurs fois sur chaque machine et faire une moyenne, ou noter la médiane.

## Important

- Assurez-vous d'utiliser la **même version de Python** et les **mêmes versions des bibliothèques** (NumPy) sur les deux machines pour une comparaison équitable. L'utilisation d'environnements virtuels et de `requirements.txt` aide grandement à cela.
- Les performances d'I/O disque peuvent être influencées par l'état du disque (fragmentation sur HDD, espace libre sur SSD) et le système de fichiers.
- Branchez les ordinateurs portables sur secteur pour éviter les limitations de performance liées à la gestion de l'énergie sur batterie.
