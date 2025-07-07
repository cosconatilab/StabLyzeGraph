# StablyzeGraph

<div align="center">
  <img src="src/logo.png" alt="StablyzeGraph Logo" width="300"/>
  
  **A PyQt6-based GUI for protein engineering with Benchmarking and Screening modes**
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
  [![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
  [![License](https://img.shields.io/badge/License-GNU-yellow.svg)](LICENSE)
</div>

## 🧬 About

StablyzeGraph is a comprehensive desktop application for protein engineering that combines machine learning with an intuitive graphical interface. It provides two main modes:

- ** Benchmarking Mode**: Evaluate protein engineering models with comprehensive metrics and visualizations
- ** Screening Mode**: Generate and rank protein mutants using advanced algorithms

## ✨ Features

- **Interactive GUI**: Modern PyQt6-based interface with real-time progress tracking
- **PyTorch Integration**: Leverages PyTorch and PyTorch Geometric for deep learning
- **Comprehensive Analysis**: Detailed metrics, plots, and result visualization
- **Flexible Input**: Supports various file formats (CSV, FASTA, PDB)
- **Cross-Platform**: Works on Linux systems with full desktop integration

## 🚀 Quick Install

### Prerequisites

- Linux operating system (Ubuntu, CentOS, Fedora, etc.)
- Internet connection for downloading dependencies
- At least 5GB of free disk space

### One-Command Installation

```bash
# Download and extract the package
wget https://github.com/your-repo/stablyzegraph/releases/latest/download/stablyzegraph-installer.tar.gz
tar -xzf stablyzegraph-installer.tar.gz
cd stablyzegraph-installer

# Run the installer
./install_stablyzegraph_gui.sh
```

### Alternative: Clone from Repository

```bash
# Clone the repository
git clone https://github.com/your-repo/stablyzegraph.git
cd stablyzegraph

# Run the installer
./install_stablyzegraph_gui.sh
```

## 📋 Installation Details

The installer will automatically:

1. **🐍 Setup Python Environment**
   - Detect or install Miniconda/Anaconda
   - Create isolated Conda environment (`stablyzegraph_env`)
   - Install PyTorch with CUDA support

2. **📦 Install Dependencies**
   - PyTorch & PyTorch Geometric
   - PyQt6 for GUI
   - BioPython for sequence analysis
   - Scientific Python stack (NumPy, Pandas, Matplotlib, etc.)

3. **🖥️ System Integration**
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
~/.local/share/stablyzegraph/stablyzegraph_launcher.sh
```

Or use the desktop shortcut/application menu entry.

### Benchmarking Mode

1. **Input Files Required:**
   - Active Sequences (CSV)
   - Inactive Sequences (CSV)
   - Wild Type (FASTA)
   - PDB Structure
   - Properties (CSV)

2. **Configure Parameters:**
   - Hidden Dimension: 128 (default)
   - Dropout Rate: 0.25 (default)
   - Learning Rate: 0.00001 (default)
   - Max Epochs: 1000 (default)

3. **Run Analysis:**
   - Click "Run Benchmarking"
   - Monitor progress in real-time
   - View results: metrics, plots, confusion matrix

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

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, Fedora 30+)
- **RAM**: 4GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: CUDA-compatible GPU (optional, for acceleration)

### Required System Packages
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install build-essential wget curl

# CentOS/RHEL
sudo yum groupinstall 'Development Tools' && sudo yum install wget curl

# Fedora
sudo dnf groupinstall 'Development Tools' && sudo dnf install wget curl
```

## 🗑️ Uninstallation

To remove StablyzeGraph completely:

```bash
# Run the uninstaller (path depends on installation choice)
~/.local/share/stablyzegraph/uninstall_stablyzegraph.sh

# Or for system-wide installation
/opt/stablyzegraph/uninstall_stablyzegraph.sh
```

To also remove the Conda environment:
```bash
conda env remove -n stablyzegraph_env
```

## 🐛 Troubleshooting

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

**Permission denied errors:**
```bash
# Ensure installer script is executable
chmod +x install_stablyzegraph_gui.sh
```

### Getting Help

1. Check the installation logs for detailed error messages
2. Ensure all system requirements are met
3. Try running the installer with verbose output:
   ```bash
   bash -x install_stablyzegraph_gui.sh
   ```

## 📁 Project Structure

```
stablyzegraph-installer/
├── install_stablyzegraph_gui.sh    # Main installer script
├── README.md                       # This file
├── NSE                         # nse file
└── src/                           # Source files
    ├── main.py                    # Main application
    ├── Benchmarking.py           # Benchmarking module
    ├── Screening.py              # Screening module
    ├── requirements.txt          # Python dependencies
    └── logo.png                  # Application logo
```

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Made with ❤️ for the protein engineering community</strong>
</div>
