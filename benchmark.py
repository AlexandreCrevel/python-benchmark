# benchmark.py
import time
import math
import os
import random
import string
import numpy as np
import hashlib
import zlib

# --- Configuration des Benchmarks ---
# Vous pouvez ajuster ces valeurs si un test est trop rapide ou trop lent
FIB_NUMBER = 40          # Calcul Fibonacci (CPU intensif, appels récursifs)
PRIME_LIMIT = 1500000     # Calcul de nombres premiers (CPU intensif, boucles)
MATRIX_SIZE = 10000       # Multiplication de matrices (CPU/SIMD, utilise NumPy optimisé)
LIST_SIZE = 20 * 1000 * 1000 # Allocation et manipulation de liste (Mémoire/CPU)
LARGE_FILE_SIZE_MB = 2560 # Taille du fichier pour test I/O disque (en Mo)
NUM_SMALL_FILES = 10000   # Nombre de petits fichiers pour test I/O disque
SMALL_FILE_SIZE_KB = 40   # Taille des petits fichiers (en Ko)
HASH_DATA_SIZE_MB = 1000  # Taille des données à hasher (CPU/Mémoire)
COMPRESS_DATA_SIZE_MB = 500 # Taille des données à compresser (CPU)

TEMP_DIR = "benchmark_temp" # Dossier temporaire pour les tests de fichiers

# --- Fonctions de Benchmark ---

def benchmark_fibonacci(n):
    """Calcule le n-ième nombre de Fibonacci de manière récursive (inefficace exprès)."""
    def fib(x):
        if x <= 1:
            return x
        else:
            return fib(x-1) + fib(x-2)
    start_time = time.perf_counter()
    result = fib(n)
    end_time = time.perf_counter()
    print(f"  -> Fibonacci({n}) = {result} (peut être différent si n > ~90 à cause des floats)")
    return end_time - start_time

def benchmark_primes(limit):
    """Trouve les nombres premiers jusqu'à une limite donnée (méthode simple)."""
    start_time = time.perf_counter()
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    end_time = time.perf_counter()
    print(f"  -> Trouvé {len(primes)} nombres premiers jusqu'à {limit}")
    return end_time - start_time

def benchmark_matrix_multiplication(size):
    """Effectue une multiplication de matrices carrées avec NumPy."""
    start_time = time.perf_counter()
    # Crée deux matrices aléatoires
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    # Multiplication
    result_matrix = np.dot(matrix_a, matrix_b)
    end_time = time.perf_counter()
    print(f"  -> Multiplication de matrices {size}x{size} terminée. Somme de la résultante : {np.sum(result_matrix):.2f}")
    return end_time - start_time

def benchmark_list_operations(size):
    """Crée une grande liste d'entiers et la trie."""
    start_time = time.perf_counter()
    # Création d'une grande liste
    my_list = [random.randint(0, size) for _ in range(size)]
    # Opération (tri) pour utiliser la mémoire et le CPU
    my_list.sort()
    end_time = time.perf_counter()
    print(f"  -> Création et tri d'une liste de {size} entiers terminé.")
    return end_time - start_time

def benchmark_large_file_io(file_size_mb, chunk_size=1024*1024):
    """Écrit puis lit un gros fichier temporaire."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    file_path = os.path.join(TEMP_DIR, "large_temp_file.bin")
    file_size_bytes = file_size_mb * 1024 * 1024
    data_chunk = os.urandom(chunk_size) # 1 Mo de données aléatoires

    # --- Écriture ---
    start_time_write = time.perf_counter()
    bytes_written = 0
    try:
        with open(file_path, 'wb') as f:
            while bytes_written < file_size_bytes:
                write_size = min(chunk_size, file_size_bytes - bytes_written)
                if write_size < chunk_size: # Ajuster le dernier chunk
                     data_chunk = os.urandom(write_size)
                f.write(data_chunk[:write_size])
                bytes_written += write_size
            f.flush() # Force l'écriture sur le disque (peut varier selon l'OS)
            os.fsync(f.fileno()) # Tente de forcer la synchronisation physique
    except Exception as e:
        print(f"  Erreur pendant l'écriture: {e}")
        # Cleanup en cas d'erreur
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, None # Indique une erreur
    end_time_write = time.perf_counter()
    write_duration = end_time_write - start_time_write
    print(f"  -> Écrit {file_size_mb} Mo sur le disque.")

    # --- Lecture ---
    start_time_read = time.perf_counter()
    bytes_read = 0
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                # On pourrait faire quelque chose avec le chunk ici pour simuler un traitement
    except Exception as e:
        print(f"  Erreur pendant la lecture: {e}")
         # Cleanup en cas d'erreur
        if os.path.exists(file_path):
            os.remove(file_path)
        return write_duration, None # Indique une erreur

    end_time_read = time.perf_counter()
    read_duration = end_time_read - start_time_read
    print(f"  -> Lu {bytes_read / (1024*1024):.2f} Mo depuis le disque.")

    # --- Nettoyage ---
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"  Avertissement: impossible de supprimer le fichier temporaire {file_path}: {e}")


    return write_duration, read_duration

def benchmark_small_files_io(num_files, file_size_kb):
    """Crée, écrit et lit de nombreux petits fichiers."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    file_paths = [os.path.join(TEMP_DIR, f"small_file_{i}.txt") for i in range(num_files)]
    file_size_bytes = file_size_kb * 1024
    # Générer des données aléatoires une seule fois
    small_data = ''.join(random.choices(string.ascii_letters + string.digits, k=file_size_bytes)).encode('utf-8')

    # --- Écriture ---
    start_time_write = time.perf_counter()
    try:
        for file_path in file_paths:
            with open(file_path, 'wb') as f:
                f.write(small_data)
        # On pourrait tenter fsync sur le dossier sous Linux/Mac mais c'est complexe
    except Exception as e:
        print(f"  Erreur pendant l'écriture des petits fichiers: {e}")
        # Cleanup partiel
        for fp in file_paths:
            if os.path.exists(fp): os.remove(fp)
        return None, None
    end_time_write = time.perf_counter()
    write_duration = end_time_write - start_time_write
    print(f"  -> Écrit {num_files} fichiers de {file_size_kb} Ko.")

    # --- Lecture ---
    start_time_read = time.perf_counter()
    total_bytes_read = 0
    try:
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                content = f.read()
                total_bytes_read += len(content)
    except Exception as e:
        print(f"  Erreur pendant la lecture des petits fichiers: {e}")
         # Cleanup partiel
        for fp in file_paths:
            if os.path.exists(fp): os.remove(fp)
        return write_duration, None
    end_time_read = time.perf_counter()
    read_duration = end_time_read - start_time_read
    print(f"  -> Lu {total_bytes_read / (1024*1024):.2f} Mo depuis {num_files} petits fichiers.")

    # --- Nettoyage ---
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"  Avertissement: problème lors du nettoyage des petits fichiers: {e}")

    return write_duration, read_duration

def benchmark_hashing(data_size_mb):
    """Génère des données aléatoires et calcule leur hash SHA-256."""
    start_time = time.perf_counter()
    data_size_bytes = data_size_mb * 1024 * 1024
    # Générer les données (peut consommer de la mémoire)
    random_data = os.urandom(data_size_bytes)
    # Calculer le hash
    sha256_hash = hashlib.sha256(random_data).hexdigest()
    end_time = time.perf_counter()
    print(f"  -> Hash SHA-256 de {data_size_mb} Mo calculé: {sha256_hash[:10]}...")
    return end_time - start_time

def benchmark_compression(data_size_mb):
    """Génère des données aléatoires et les compresse avec zlib."""
    start_time_compress = time.perf_counter()
    data_size_bytes = data_size_mb * 1024 * 1024
    # Générer les données
    original_data = os.urandom(data_size_bytes)
    # Compression
    compressed_data = zlib.compress(original_data, level=6) # Niveau de compression par défaut
    end_time_compress = time.perf_counter()
    compress_duration = end_time_compress - start_time_compress
    original_size = len(original_data)
    compressed_size = len(compressed_data)
    ratio = compressed_size / original_size if original_size > 0 else 0
    print(f"  -> Compression de {data_size_mb} Mo terminée. Ratio: {ratio:.2f} ({compressed_size} bytes)")

    # Décompression (pour vérifier et benchmarker aussi)
    start_time_decompress = time.perf_counter()
    decompressed_data = zlib.decompress(compressed_data)
    end_time_decompress = time.perf_counter()
    decompress_duration = end_time_decompress - start_time_decompress
    print(f"  -> Décompression de {compressed_size / (1024*1024):.2f} Mo terminée.")
    # Vérification simple
    if len(decompressed_data) != original_size:
         print("  -> ERREUR: La taille décompressée ne correspond pas !")


    return compress_duration, decompress_duration

# --- Exécution des Benchmarks ---

if __name__ == "__main__":
    results = {}
    print("Démarrage des benchmarks Python...")
    print("-" * 30)
    print(f"Configuration : Python {'.'.join(map(str, os.sys.version_info[:3]))}, NumPy {np.__version__}")
    print("-" * 30)

    # 1. Calcul CPU - Fibonacci
    print(f"[1] Benchmark CPU: Calcul de Fibonacci({FIB_NUMBER})")
    results['fibonacci'] = benchmark_fibonacci(FIB_NUMBER)
    print(f"Temps écoulé: {results['fibonacci']:.4f} secondes\n")

    # 2. Calcul CPU - Nombres Premiers
    print(f"[2] Benchmark CPU: Calcul des nombres premiers jusqu'à {PRIME_LIMIT}")
    results['primes'] = benchmark_primes(PRIME_LIMIT)
    print(f"Temps écoulé: {results['primes']:.4f} secondes\n")

    # 3. Calcul CPU/SIMD - Multiplication de Matrices (NumPy)
    print(f"[3] Benchmark CPU/SIMD: Multiplication de matrices ({MATRIX_SIZE}x{MATRIX_SIZE})")
    results['matrix_multiply'] = benchmark_matrix_multiplication(MATRIX_SIZE)
    print(f"Temps écoulé: {results['matrix_multiply']:.4f} secondes\n")

    # 4. Mémoire/CPU - Opérations sur Listes
    print(f"[4] Benchmark Mémoire/CPU: Création et tri d'une liste ({LIST_SIZE} éléments)")
    results['list_operations'] = benchmark_list_operations(LIST_SIZE)
    print(f"Temps écoulé: {results['list_operations']:.4f} secondes\n")

    # 5. I/O Disque - Gros Fichier
    print(f"[5] Benchmark I/O Disque: Écriture/Lecture d'un fichier de {LARGE_FILE_SIZE_MB} Mo")
    write_time, read_time = benchmark_large_file_io(LARGE_FILE_SIZE_MB)
    if write_time is not None:
        results['large_file_write'] = write_time
        print(f"Temps d'écriture : {results['large_file_write']:.4f} secondes")
    if read_time is not None:
        results['large_file_read'] = read_time
        print(f"Temps de lecture  : {results['large_file_read']:.4f} secondes\n")
    else:
         print("Le test d'I/O sur gros fichier a échoué.\n")


    # 6. I/O Disque - Petits Fichiers
    print(f"[6] Benchmark I/O Disque: Écriture/Lecture de {NUM_SMALL_FILES} fichiers de {SMALL_FILE_SIZE_KB} Ko")
    write_time_small, read_time_small = benchmark_small_files_io(NUM_SMALL_FILES, SMALL_FILE_SIZE_KB)
    if write_time_small is not None:
        results['small_files_write'] = write_time_small
        print(f"Temps d'écriture : {results['small_files_write']:.4f} secondes")
    if read_time_small is not None:
        results['small_files_read'] = read_time_small
        print(f"Temps de lecture  : {results['small_files_read']:.4f} secondes\n")
    else:
         print("Le test d'I/O sur petits fichiers a échoué.\n")


    # 7. CPU/Mémoire - Hashing
    print(f"[7] Benchmark CPU/Mémoire: Hash SHA-256 de {HASH_DATA_SIZE_MB} Mo de données")
    results['hashing'] = benchmark_hashing(HASH_DATA_SIZE_MB)
    print(f"Temps écoulé: {results['hashing']:.4f} secondes\n")

    # 8. CPU - Compression/Décompression
    print(f"[8] Benchmark CPU: Compression/Décompression zlib de {COMPRESS_DATA_SIZE_MB} Mo de données")
    compress_time, decompress_time = benchmark_compression(COMPRESS_DATA_SIZE_MB)
    if compress_time is not None:
        results['compression'] = compress_time
        print(f"Temps de compression   : {results['compression']:.4f} secondes")
    if decompress_time is not None:
        results['decompression'] = decompress_time
        print(f"Temps de décompression : {results['decompression']:.4f} secondes\n")
    else:
        print("Le test de compression/décompression a échoué.\n")


    # --- Nettoyage final du dossier temporaire ---
    if os.path.exists(TEMP_DIR):
        try:
            # Supprimer les fichiers restants s'il y en a
            for item in os.listdir(TEMP_DIR):
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            # Supprimer le dossier lui-même
            os.rmdir(TEMP_DIR)
            print(f"Dossier temporaire '{TEMP_DIR}' nettoyé.")
        except Exception as e:
            print(f"Avertissement: Impossible de nettoyer complètement le dossier temporaire '{TEMP_DIR}': {e}")

    # --- Affichage du Résumé ---
    print("=" * 30)
    print("RÉSUMÉ DES RÉSULTATS (en secondes, plus bas = meilleur)")
    print("-" * 30)
    for name, duration in results.items():
        if duration is not None:
             print(f"{name:<25}: {duration:.4f}")
        else:
             print(f"{name:<25}: ÉCHEC")
    print("=" * 30)