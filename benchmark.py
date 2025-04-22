# benchmark.py
import time
import math
import os
import random
import string
import numpy as np
import hashlib
import zlib
from typing import Dict, Optional, List
from io import StringIO
import sys
from datetime import datetime
import platform
import psutil
import cpuinfo
import GPUtil

# --- Benchmark Configuration ---
# You can adjust these values if a test is too fast or too slow
FIB_NUMBER = 40          # Fibonacci calculation (CPU intensive, recursive calls)
PRIME_LIMIT = 1500000     # Prime numbers calculation (CPU intensive, loops)
MATRIX_SIZE = 10000       # Matrix multiplication (CPU/SIMD, uses optimized NumPy)
LIST_SIZE = 20 * 1000 * 1000 # List allocation and manipulation (Memory/CPU)
LARGE_FILE_SIZE_MB = 2560 # File size for disk I/O test (in MB)
NUM_SMALL_FILES = 10000   # Number of small files for disk I/O test
SMALL_FILE_SIZE_KB = 40   # Size of small files (in KB)
HASH_DATA_SIZE_MB = 1000  # Size of data to hash (CPU/Memory)
COMPRESS_DATA_SIZE_MB = 500 # Size of data to compress (CPU)

TEMP_DIR = "benchmark_temp" # Temporary directory for file tests

def get_system_info() -> Dict[str, str]:
    """Get detailed system information."""
    info = {
        "python": '.'.join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "machine": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpuinfo.get_cpu_info()['brand_raw'],
        "ram": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "python_bits": f"{platform.architecture()[0]}",
    }
    
    # Get GPU information if available
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            info['gpu'] = f"{gpus[0].name} ({gpus[0].memoryTotal}MB)"
        else:
            info['gpu'] = "No GPU detected"
    except:
        info['gpu'] = "Unable to detect GPU"
    
    return info

# --- Benchmark Functions ---

def benchmark_fibonacci(n):
    """Calculate the nth Fibonacci number recursively (intentionally inefficient)."""
    def fib(x):
        if x <= 1:
            return x
        else:
            return fib(x-1) + fib(x-2)
    start_time = time.perf_counter()
    result = fib(n)
    end_time = time.perf_counter()
    print(f"  -> Fibonacci({n}) = {result} (may differ if n > ~90 due to floats)")
    return end_time - start_time

def benchmark_primes(limit):
    """Find prime numbers up to a given limit (simple method)."""
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
    print(f"  -> Found {len(primes)} prime numbers up to {limit}")
    return end_time - start_time

def benchmark_matrix_multiplication(size):
    """Perform square matrix multiplication with NumPy."""
    start_time = time.perf_counter()
    # Create two random matrices
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    # Multiplication
    result_matrix = np.dot(matrix_a, matrix_b)
    end_time = time.perf_counter()
    print(f"  -> Matrix multiplication {size}x{size} completed. Sum of result: {np.sum(result_matrix):.2f}")
    return end_time - start_time

def benchmark_list_operations(size):
    """Create a large list of integers and sort it."""
    start_time = time.perf_counter()
    # Create a large list
    my_list = [random.randint(0, size) for _ in range(size)]
    # Operation (sort) to use memory and CPU
    my_list.sort()
    end_time = time.perf_counter()
    print(f"  -> Creation and sorting of a list of {size} integers completed.")
    return end_time - start_time

def benchmark_large_file_io(file_size_mb, chunk_size=1024*1024):
    """Write then read a large temporary file."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    file_path = os.path.join(TEMP_DIR, "large_temp_file.bin")
    file_size_bytes = file_size_mb * 1024 * 1024
    data_chunk = os.urandom(chunk_size) # 1 Mo of random data

    # --- Writing ---
    start_time_write = time.perf_counter()
    bytes_written = 0
    try:
        with open(file_path, 'wb') as f:
            while bytes_written < file_size_bytes:
                write_size = min(chunk_size, file_size_bytes - bytes_written)
                if write_size < chunk_size: # Adjust last chunk
                     data_chunk = os.urandom(write_size)
                f.write(data_chunk[:write_size])
                bytes_written += write_size
            f.flush() # Force disk write (may vary by OS)
            os.fsync(f.fileno()) # Try to force physical sync
    except Exception as e:
        print(f"  Error during writing: {e}")
        # Cleanup in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, None
    end_time_write = time.perf_counter()
    write_duration = end_time_write - start_time_write
    print(f"  -> Wrote {file_size_mb} Mo to disk.")

    # --- Reading ---
    start_time_read = time.perf_counter()
    bytes_read = 0
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                # We could do something with the chunk here to simulate processing
    except Exception as e:
        print(f"  Error during reading: {e}")
        # Cleanup in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return write_duration, None # Indicate an error

    end_time_read = time.perf_counter()
    read_duration = end_time_read - start_time_read
    print(f"  -> Read {bytes_read / (1024*1024):.2f} Mo from disk.")

    # --- Nettoyage ---
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"  Warning: Could not delete temporary file {file_path}: {e}")


    return write_duration, read_duration

def benchmark_small_files_io(num_files, file_size_kb):
    """Create, write and read many small files."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    file_paths = [os.path.join(TEMP_DIR, f"small_file_{i}.txt") for i in range(num_files)]
    file_size_bytes = file_size_kb * 1024
    # Generate random data once
    small_data = ''.join(random.choices(string.ascii_letters + string.digits, k=file_size_bytes)).encode('utf-8')

    # --- Writing ---
    start_time_write = time.perf_counter()
    try:
        for file_path in file_paths:
            with open(file_path, 'wb') as f:
                f.write(small_data)
        # We could try fsync on the directory under Linux/Mac but it's complex
    except Exception as e:
        print(f"  Error during writing small files: {e}")
        # Cleanup partial
        for fp in file_paths:
            if os.path.exists(fp): os.remove(fp)
        return None, None
    end_time_write = time.perf_counter()
    write_duration = end_time_write - start_time_write
    print(f"  -> Wrote {num_files} files of {file_size_kb} Ko.")

    # --- Reading ---
    start_time_read = time.perf_counter()
    total_bytes_read = 0
    try:
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                content = f.read()
                total_bytes_read += len(content)
    except Exception as e:
        print(f"  Error during reading small files: {e}")
        # Cleanup partial
        for fp in file_paths:
            if os.path.exists(fp): os.remove(fp)
        return write_duration, None
    end_time_read = time.perf_counter()
    read_duration = end_time_read - start_time_read
    print(f"  -> Read {total_bytes_read / (1024*1024):.2f} Mo from {num_files} small files.")

    # --- Cleanup ---
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"  Warning: Could not clean up small files: {e}")

    return write_duration, read_duration

def benchmark_hashing(data_size_mb):
    """Generate random data and calculate its SHA-256 hash."""
    start_time = time.perf_counter()
    data_size_bytes = data_size_mb * 1024 * 1024
    # Générer les données (peut consommer de la mémoire)
    random_data = os.urandom(data_size_bytes)
    # Calculer le hash
    sha256_hash = hashlib.sha256(random_data).hexdigest()
    end_time = time.perf_counter()
    print(f"  -> SHA-256 hash of {data_size_mb} Mo calculated: {sha256_hash[:10]}...")
    return end_time - start_time

def benchmark_compression(data_size_mb):
    """Generate random data and compress it with zlib."""
    start_time_compress = time.perf_counter()
    data_size_bytes = data_size_mb * 1024 * 1024
    # Generate the data
    original_data = os.urandom(data_size_bytes)
    # Compress
    compressed_data = zlib.compress(original_data, level=6) # Default compression level
    end_time_compress = time.perf_counter()
    compress_duration = end_time_compress - start_time_compress
    original_size = len(original_data)
    compressed_size = len(compressed_data)
    ratio = compressed_size / original_size if original_size > 0 else 0
    print(f"  -> Compression of {data_size_mb} Mo completed. Ratio: {ratio:.2f} ({compressed_size} bytes)")

    # Decompress (for verification and benchmarking)
    start_time_decompress = time.perf_counter()
    decompressed_data = zlib.decompress(compressed_data)
    end_time_decompress = time.perf_counter()
    decompress_duration = end_time_decompress - start_time_decompress
    print(f"  -> Decompression of {compressed_size / (1024*1024):.2f} Mo completed.")
    # Simple verification
    if len(decompressed_data) != original_size:
        print("  -> Error: Decompressed size does not match!")

    return compress_duration, decompress_duration

# --- Execution of Benchmarks ---

def generate_markdown_report(
    results: Dict[str, Optional[float]],
    config: Dict[str, str],
    logs: List[str],
    output_path: str
) -> None:
    """
    Generate a markdown benchmark report.
    Args:
        results: dict of test name to duration
        config: dict of configuration strings
        logs: list of log strings for each benchmark
        output_path: path to write the .md file
    """
    # Header
    md = [
        f"# Python Benchmark Results – {config.get('machine', 'Unknown Machine')}",
        "",
        "## System Information",
        "",
        f"- **Operating System**: {config['os']}",
        f"- **CPU**: {config['cpu']}",
        f"- **RAM**: {config['ram']}",
        f"- **GPU**: {config['gpu']}",
        f"- **Python**: {config['python']} ({config['python_bits']})",
        f"- **NumPy**: {config['numpy']}",
        "",
        "---",
        "",
        "## Detailed Benchmarks",
        ""
    ]
    md.extend(logs)
    md.append("\n---\n")
    md.append(f"> Temporary directory `{config.get('temp_dir', 'benchmark_temp')}` cleaned.")
    md.append("\n---\n")
    md.append("## Results Summary (seconds, lower = better)\n")
    md.append("| Test                | Time (s) |\n|---------------------|-----------|")
    labels = {
        "fibonacci": "Fibonacci",
        "primes": "Primes",
        "matrix_multiply": "Matrix Multiply",
        "list_operations": "List Operations",
        "large_file_write": "Large File Write",
        "large_file_read": "Large File Read",
        "small_files_write": "Small Files Write",
        "small_files_read": "Small Files Read",
        "hashing": "Hashing",
        "compression": "Compression",
        "decompression": "Decompression"
    }
    for key, label in labels.items():
        val = results.get(key)
        # Format the value conditionally first, then align the resulting string
        value_str = f"{val:.4f}" if val is not None else "FAILURE"
        md.append(f"| {label:<19} | {value_str:<9} |")
    md.append("\n---\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

if __name__ == "__main__":
    results = {}
    logs: List[str] = []
    config = get_system_info()
    def log_md(title: str, content: str) -> None:
        content = content.replace("Temps écoulé:", "Time elapsed:")
        content = content.replace("Temps d'écriture :", "Write time:")
        content = content.replace("Temps de lecture :", "Read time:")
        content = content.replace("Temps de compression :", "Compression time:")
        content = content.replace("Temps de décompression :", "Decompression time:")
        content = content.replace("secondes", "seconds")
        content = content.replace("Écrit", "Wrote")
        content = content.replace("Lu", "Read")
        content = content.replace("fichiers de", "files of")
        content = content.replace("petits fichiers", "small files")
        content = content.replace("depuis", "from")
        content = content.replace("sur le disque", "to disk")
        content = content.replace("Hash SHA-256 de", "SHA-256 hash of")
        content = content.replace("calculé:", "calculated:")
        content = content.replace("Mo", "MB")
        logs.append(f"### {title}\n```text\n{content}\n```")
    print("Starting Python benchmarks...")
    print("-" * 50)
    print("System Configuration:")
    print(f"OS: {config['os']}")
    print(f"CPU: {config['cpu']}")
    print(f"RAM: {config['ram']}")
    print(f"GPU: {config['gpu']}")
    print(f"Python: {config['python']} ({config['python_bits']})")
    print(f"NumPy: {config['numpy']}")
    print("-" * 50)

    # 1. CPU - Fibonacci
    print(f"[1] CPU Benchmark: Fibonacci({FIB_NUMBER})")
    buf = StringIO()
    sys.stdout = buf
    results['fibonacci'] = benchmark_fibonacci(FIB_NUMBER)
    sys.stdout = sys.__stdout__
    log_md("1. CPU - Fibonacci(40)", buf.getvalue() + f"Time elapsed: {results['fibonacci']:.4f} seconds")
    print(f"Time elapsed: {results['fibonacci']:.4f} seconds\n")

    # 2. CPU - Prime Numbers
    print(f"[2] CPU Benchmark: Prime numbers up to {PRIME_LIMIT}")
    buf = StringIO()
    sys.stdout = buf
    results['primes'] = benchmark_primes(PRIME_LIMIT)
    sys.stdout = sys.__stdout__
    log_md("2. CPU - Prime Numbers up to 1,500,000", buf.getvalue() + f"Time elapsed: {results['primes']:.4f} seconds")
    print(f"Time elapsed: {results['primes']:.4f} seconds\n")

    # 3. CPU/SIMD - Matrix Multiplication
    print(f"[3] CPU/SIMD Benchmark: Matrix multiplication ({MATRIX_SIZE}x{MATRIX_SIZE})")
    buf = StringIO()
    sys.stdout = buf
    results['matrix_multiply'] = benchmark_matrix_multiplication(MATRIX_SIZE)
    sys.stdout = sys.__stdout__
    log_md("3. CPU/SIMD - Matrix Multiplication (10000x10000)", buf.getvalue() + f"Time elapsed: {results['matrix_multiply']:.4f} seconds")
    print(f"Time elapsed: {results['matrix_multiply']:.4f} seconds\n")

    # 4. Memory/CPU - List Operations
    print(f"[4] Memory/CPU Benchmark: List creation and sorting ({LIST_SIZE:,} elements)")
    buf = StringIO()
    sys.stdout = buf
    results['list_operations'] = benchmark_list_operations(LIST_SIZE)
    sys.stdout = sys.__stdout__
    log_md("4. Memory/CPU - List Creation and Sorting (20,000,000 elements)", buf.getvalue() + f"Time elapsed: {results['list_operations']:.4f} seconds")
    print(f"Time elapsed: {results['list_operations']:.4f} seconds\n")

    # 5. Disk I/O - Large File
    print(f"[5] Disk I/O Benchmark: Writing/Reading a {LARGE_FILE_SIZE_MB} MB file")
    buf = StringIO()
    sys.stdout = buf
    write_time, read_time = benchmark_large_file_io(LARGE_FILE_SIZE_MB)
    sys.stdout = sys.__stdout__
    if write_time is not None:
        results['large_file_write'] = write_time
    if read_time is not None:
        results['large_file_read'] = read_time
    log_md("5. Disk I/O - Writing/Reading a 2560 MB file", buf.getvalue() + (f"Write time: {results.get('large_file_write', 0):.4f} seconds\nRead time: {results.get('large_file_read', 0):.4f} seconds" if write_time and read_time else "FAILURE"))
    if write_time is not None:
        print(f"Write time: {results['large_file_write']:.4f} seconds")
    if read_time is not None:
        print(f"Read time: {results['large_file_read']:.4f} seconds\n")
    else:
         print("Large file I/O test failed.\n")

    # 6. Disk I/O - Small Files
    print(f"[6] Disk I/O Benchmark: Writing/Reading {NUM_SMALL_FILES} files of {SMALL_FILE_SIZE_KB} KB")
    buf = StringIO()
    sys.stdout = buf
    write_time_small, read_time_small = benchmark_small_files_io(NUM_SMALL_FILES, SMALL_FILE_SIZE_KB)
    sys.stdout = sys.__stdout__
    if write_time_small is not None:
        results['small_files_write'] = write_time_small
    if read_time_small is not None:
        results['small_files_read'] = read_time_small
    log_md("6. Disk I/O - Writing/Reading 10,000 files of 40 KB", buf.getvalue() + (f"Write time: {results.get('small_files_write', 0):.4f} seconds\nRead time: {results.get('small_files_read', 0):.4f} seconds" if write_time_small and read_time_small else "FAILURE"))
    if write_time_small is not None:
        print(f"Write time: {results['small_files_write']:.4f} seconds")
    if read_time_small is not None:
        print(f"Read time: {results['small_files_read']:.4f} seconds\n")
    else:
         print("Small files I/O test failed.\n")

    # 7. CPU/Memory - Hashing
    print(f"[7] CPU/Memory Benchmark: SHA-256 hash of {HASH_DATA_SIZE_MB} MB of data")
    buf = StringIO()
    sys.stdout = buf
    results['hashing'] = benchmark_hashing(HASH_DATA_SIZE_MB)
    sys.stdout = sys.__stdout__
    log_md("7. CPU/Memory - SHA-256 hash of 1000 MB of data", buf.getvalue() + f"Time elapsed: {results['hashing']:.4f} seconds")
    print(f"Time elapsed: {results['hashing']:.4f} seconds\n")

    # 8. CPU - Compression/Decompression
    print(f"[8] CPU Benchmark: zlib Compression/Decompression of {COMPRESS_DATA_SIZE_MB} MB of data")
    buf = StringIO()
    sys.stdout = buf
    compress_time, decompress_time = benchmark_compression(COMPRESS_DATA_SIZE_MB)
    sys.stdout = sys.__stdout__
    if compress_time is not None:
        results['compression'] = compress_time
    if decompress_time is not None:
        results['decompression'] = decompress_time
    log_md("8. CPU - zlib Compression/Decompression of 500 MB", buf.getvalue() + (f"Compression time: {results.get('compression', 0):.4f} seconds\nDecompression time: {results.get('decompression', 0):.4f} seconds" if compress_time and decompress_time else "FAILURE"))
    if compress_time is not None:
        print(f"Compression time: {results['compression']:.4f} seconds")
    if decompress_time is not None:
        print(f"Decompression time: {results['decompression']:.4f} seconds\n")
    else:
        print("Compression/Decompression test failed.\n")


    # --- Final cleanup of the temporary directory ---
    if os.path.exists(TEMP_DIR):
        try:
            # Remove any remaining files
            for item in os.listdir(TEMP_DIR):
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            # Remove the directory itself
            os.rmdir(TEMP_DIR)
            print(f"Temporary directory '{TEMP_DIR}' cleaned.")
        except Exception as e:
            print(f"Warning: Could not completely clean the temporary directory '{TEMP_DIR}': {e}")

    # --- Génération du rapport Markdown ---
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"benchmark_{now}.md")
    generate_markdown_report(results, config, logs, output_path)
    print(f"\nMarkdown report generated in: {output_path}\n")

    # --- Affichage du Résumé ---
    print("=" * 30)
    print("RESULTS SUMMARY (in seconds, lower = better)")
    print("-" * 30)
    for name, duration in results.items():
        if duration is not None:
             print(f"{name:<25}: {duration:.4f}")
        else:
             print(f"{name:<25}: FAILURE")
    print("=" * 30)