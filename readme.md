# Simple Python Benchmark Project

This project provides a set of Python scripts to evaluate and compare the performance of different machines (for example, a Windows PC and a MacBook) on various computational tasks.

## Prerequisites

- **Python 3**: Make sure Python 3 (version 3.7 or higher recommended) is installed on both machines.
- **pip**: The Python package manager, usually included with Python.

## Installation

1.  **Clone or copy this project** on both machines you want to compare. Place it in a directory of your choice.
2.  **Open a terminal or command prompt** in the `python-benchmark` folder.
3.  **(Recommended)** Create and activate a virtual environment:

    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows (cmd)
    python -m venv venv
    venv\Scripts\activate

    # On Windows (PowerShell)
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Benchmarks

1.  **Make sure no other resource-intensive applications** are running in the background on the machine to get more consistent results.
2.  **Run the main script** from the terminal (make sure your virtual environment is activated if you're using one):
    ```bash
    python benchmark.py
    ```
3.  The script will sequentially run each benchmark and display the time taken for each. A final summary will also be displayed.
4.  **Repeat the operation** on the other machine using **exactly the same code and library versions** (thanks to `requirements.txt` and the virtual environment).

## The Benchmarks

- **Fibonacci (CPU)**: Tests raw CPU performance on recursive function calls (single-thread).
- **Prime Numbers (CPU)**: Tests CPU performance on intensive calculations and loops (single-thread).
- **Matrix Multiplication (CPU/SIMD)**: Uses NumPy to test numerical computation capabilities, often optimized via SIMD and underlying C/Fortran libraries. May use multiple cores depending on NumPy/BLAS installation.
- **List Operations (Memory/CPU)**: Measures the time needed to allocate a large amount of memory for a Python list and perform an operation on it (sorting).
- **Disk I/O - Large File**: Tests sequential read and write speed on HDD/SSD.
- **Disk I/O - Small Files**: Tests filesystem performance for creating, writing, and reading many small files, which is common during project builds or cache management.
- **Hashing (CPU/Memory)**: Measures the speed of calculating a SHA-256 hash on a quantity of data in memory.
- **Compression/Decompression (CPU)**: Tests CPU performance on compression/decompression algorithms (zlib).

## Interpreting Results

- Compare the times (in seconds) obtained for each benchmark between the two machines.
- **A shorter time indicates better performance** for that specific test.
- Note the relative differences. A machine might excel in pure CPU but be slower in disk I/O, for example.
- For more reliable results, you can run the script multiple times on each machine and take an average, or note the median.

## Important

- Make sure to use the **same Python version** and the **same library versions** (NumPy) on both machines for a fair comparison. Using virtual environments and `requirements.txt` greatly helps with this.
- Disk I/O performance can be influenced by disk state (fragmentation on HDD, free space on SSD) and the filesystem.
- Plug in laptops to AC power to avoid performance limitations related to battery power management.
