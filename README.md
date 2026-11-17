# StabLyzeGraph

<div align="center">
  <img src="src/logo.png" alt="StablyzeGraph Logo" width="300"/>
  
  **StabLyzeGraph: High-Throughput Screening of Combinatorial Mutations Using Graph Neural Networks**
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
  [![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
  [![License](https://img.shields.io/badge/License-GNU-yellow.svg)](LICENSE)
</div>

##  About

**StabLyzeGraph** is a comprehensive computational framework for accelerated protein stability engineering, tackling the challenges of vast mutational space and complex protein interactions using graph neural networks (GNNs). StabLyzeGraph integrates diverse protein features (physicochemical properties, evolutionary conservation, and 3D structure coordinates) to predict the impact of mutations, identify beneficial combinations, and streamline the discovery of highly stable proteins. It provides two main modes:

- **Benchmarking Mode**: In benchmarking mode, most robust and more relevent GNN model is fined tuned by hyperparameter optimization.
- **Screening Mode**: In screening mode, first a library of mutants generated then screened using optimized/trained model. 

##  Features

- **Interactive GUI**: Modern PyQt6-based interface with real-time progress tracking
- **PyTorch Integration**: Leverages PyTorch and PyTorch Geometric for deep learning
- **Comprehensive Analysis**: Detailed metrics, plots, and result visualization
- **Cross-Platform**: Works on Linux systems with full desktop integration and Windows 64 bit

##  Quick Install

### Prerequisites

- Linux operating system (Ubuntu, CentOS, Fedora, etc.) or Windows 64 bit
- Internet connection for downloading dependencies
- At least 5GB of free disk space

### One-Command Installation for Linux

```bash
# Download and extract the package, then run the installer
python install_stablyzegraph_gui.py
```

### Alternative: Clone from Repository

```bash
# Clone the repository
git clone https://github.com/cosconatilab/StabLyzeGraph.git
cd StabLyzeGraph

# Run the installer
python install_stablyzegraph_gui.py
```

##  Installation Details

The installer will automatically:

1. **Setup Python Environment**
   - Detect or install Miniconda/Anaconda
   - Create isolated Conda environment (`stablyzegraph_env`)
   - Install PyTorch with CUDA support

2. **Install Dependencies**
   - PyTorch & PyTorch Geometric
   - PyQt6 for GUI
   - Clustalo
   - BioPython for sequence analysis
   - Scientific Python stack (NumPy, Pandas, Matplotlib, etc.)

3. **System Integration**
   - Create `stablyzegraph` terminal command
   - Add desktop entry and application menu integration
   - Generate desktop shortcut
   - Setup uninstaller

### Installation Options

During installation, you can choose:

- **System-wide installation** (`/opt/stablyzegraph`) - requires sudo
- **User installation** (`~/.local/share/stablyzegraph`) - recommended

##  Usage

### Launch Methods

After installation, you can launch StablyzeGraph using any of these methods:

```bash
# Terminal command (available anywhere)
stablyzegraph

# Direct launcher script
~/.local/share/stablyzegraph/stablyzegraph.sh
```

Or use the desktop shortcut/application menu entry.

### Benchmarking Mode

1. **Input Files Required:**
   - Active Sequences (CSV)
   - Inactive Sequences (CSV)
   - Wild Type (FASTA)
   - PDB Structure
   - Properties (CSV)

2. **Configure Parameters (Example):**
   - Hidden Dimension: 128 (default)
   - Dropout Rate: 0.25 (default)
   - Learning Rate: 0.00001 (default)
   - Max Epochs: 1000 (default)

3. **Run Analysis:**
   - Click "Run Benchmarking"
   - Monitor progress in real-time
   - View results: metrics, plots

### Screening Mode

1. **Input Files Required:**
   - Same as Benchmarking mode
   - Optional: Pre-trained model file

2. **Configure Parameters:**
   - Number of Mutations: 2 (default)
   - Generation Method: combinatorial (default)

3. **Generate Mutants:**
   - Click "Run Screening"
   - View ranked mutants table
   - Export results

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, Fedora 30+) or Windows 64 bit
- **RAM**: 4GB (16GB recommended)
- **Storage**: 5GB free space

### Required System Packages
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install build-essential wget curl

# CentOS/RHEL
sudo yum groupinstall 'Development Tools' && sudo yum install wget curl

# Fedora
sudo dnf groupinstall 'Development Tools' && sudo dnf install wget curl
```

## Uninstallation

To remove StablyzeGraph completely:

```bash
# Run the uninstaller (path depends on installation choice)
~/.local/share/stablyzegraph/uninstall_stablyzegraph.sh

# Or for system-wide installation
/opt/stablyzegraph/uninstall_stablyzegraph.sh
```

To also remove the Conda environment:
```bash
conda env remove -n stablyzegraph
```

## Troubleshooting

### Common Issues

**Installation fails with "conda not found":**
```bash
# The installer will automatically install Miniconda
# If issues persist, manually install conda first
```

**GUI doesn't start:**
```bash
# Check if PyQt6 is properly installed
conda activate stablyzegraph_env
python -c "import PyQt6; print('PyQt6 OK')"
```

**GUI is not opening properly:**
```bash
# Try to export QT platform manually and then try
export QT_QPA_PLATFORM=xcb
```

### Getting Help

1. Check the installation logs for detailed error messages
2. Ensure all system requirements are met
3. Try running the installer with verbose output:
   ```bash
   python install_stablyzegraph_gui.py
   ```

## 📁 Project Structure

```
stablyzegraph-installer/
├── install_stablyzegraph_gui.py    # Main installer script
├── README.md                       # This file
└── Tutorial/                       # Example input files and results for default comparison
    └── Benchmarking/               # Input and Result data files for benhcmarking analysis
    └── Screening/                  # Input and Result data files for benhcmarking analysis
└── src/                            # Source files
    ├── main.py                     # Main application
    ├── Benchmarking.py             # Benchmarking module
    ├── Screening.py                # Screening module
    ├── requirements.txt            # Python dependencies
    ├── logo.png                    # Application logo
    ├── icon.png                    # Application desktop icon
└── StabLyzeGraph_Windows_x64_Installer/    # Installation folder for Windows
    ├── StabLyzeGraph_Installer.exe         # Installer for Windows version
└── python_scripts/
    ├── Mutation_list_to_sequence_conversion.py # Module for the conversion of a mutation list into protein mutant sequences
    ├── Data_augmentation.py             # Module for augmenting negative data class (if required)
    ├── Similarity_check_BLOSUM.py       # Module to run BLOSUM similarity check
└── All_Benchmarking/               # Benchmarking Results
    ├── Data_N                     # All Datasets Directories
└── All_Datasets/                   # Curated Datasets
    ├── Data_N                     # All Datasets Directories

```

## Authors and citation

StabLyzeGraph was developed by Muhammad Waqas and Benito Natale in the Cosconati lab (University of Campania Luigi Vanvitelli), together with Michele Roggia, Prasenjit Prasad Saha, Mauro Mileni and Sandro Cosconati.

If you use StabLyzeGraph, please cite:

Waqas M, Natale B, Roggia M, Saha PP, Mileni M, Cosconati S. StabLyzeGraph: High-throughput screening of combinatorial mutations using graph neural networks. *Protein Science* 2026, 35(4), e70534. [https://doi.org/10.1002/pro.70534](https://doi.org/10.1002/pro.70534)

## License

This project is licensed under the GNU Affero General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong></strong>
</div>
