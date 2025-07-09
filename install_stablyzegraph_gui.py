#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

INSTALL_DIR = Path.home() / "StablyzeGraph"
SRC_DIR = Path(__file__).parent / "src"
REQUIREMENTS_FILE = Path(__file__).parent / "python_scripts" / "requirements.txt"
CONDA_ENV_NAME = "stablyzegraph"
ICON_FILENAME = "icon.png"
LAUNCHER_SCRIPT = INSTALL_DIR / "stablyzegraph.sh"
GLOBAL_LINK_PATH = Path.home() / ".local/bin/stablyzegraph"
DESKTOP_ENTRY_PATH = Path.home() / ".local/share/applications/stablyzegraph.desktop"
UNINSTALL_SCRIPT_PATH = INSTALL_DIR / "uninstall_stablyzegraph.sh"


def check_command(cmd):
    try:
        subprocess.run([cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False


def install_miniconda():
    print("⚠️ Conda not found. Installing Miniconda...")
    miniconda_script = Path("/tmp/miniconda.sh")
    url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    subprocess.run(["wget", "-O", str(miniconda_script), url], check=True)
    subprocess.run(["bash", str(miniconda_script), "-b", "-p", str(Path.home() / "miniconda3")], check=True)

    conda_bin = Path.home() / "miniconda3" / "bin"
    os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ["PATH"]

    subprocess.run([str(conda_bin / "conda"), "init"], check=True)
    print("✅ Miniconda installed and initialized. Please restart your terminal.")


def get_conda_python_path():
    result = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(CONDA_ENV_NAME + " "):
            env_path = line.split()[-1]
            return Path(env_path) / "bin" / "python"
    raise RuntimeError("Could not find Conda environment path.")


def create_conda_env():
    print(f"📦 Creating Conda environment '{CONDA_ENV_NAME}'...")
    subprocess.run(["conda", "create", "-y", "-n", CONDA_ENV_NAME, "python=3.10"], check=True)


def install_dependencies():
    print("📥 Installing Python dependencies...")
    if not REQUIREMENTS_FILE.exists():
        print(f"❌ {REQUIREMENTS_FILE} not found.")
        sys.exit(1)
    subprocess.run(["conda", "run", "-n", CONDA_ENV_NAME, "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True)
    subprocess.run(["conda", "run", "-n", CONDA_ENV_NAME, "pip", "install", "PyQt6", "pyqtgraph"], check=True)


def install_clustalo():
    print("🧬 Installing ClustalO...")
    try:
        subprocess.run(["conda", "run", "-n", CONDA_ENV_NAME, "conda", "install", "-y", "-c", "bioconda", "clustalo"], check=True)
        result = subprocess.run(["conda", "run", "-n", CONDA_ENV_NAME, "clustalo", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"✅ ClustalO installed: {result.stdout.decode().strip()}")
        else:
            raise Exception("ClustalO not responding.")
    except Exception as e:
        print(f"⚠️ ClustalO install failed. Please install manually:\nconda install -n {CONDA_ENV_NAME} -c bioconda clustalo\nError: {e}")


def copy_application_files():
    print(f"📂 Copying application files to {INSTALL_DIR}...")
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    if SRC_DIR.exists():
        shutil.copytree(SRC_DIR, INSTALL_DIR, dirs_exist_ok=True)

    main_py_src = Path(__file__).parent / "main.py"
    main_py_dest = INSTALL_DIR / "main.py"

    if main_py_src.exists():
        shutil.copy2(main_py_src, main_py_dest)

    python_scripts_src = Path(__file__).parent / "python_scripts"
    python_scripts_dest = INSTALL_DIR / "python_scripts"

    if python_scripts_src.exists():
        if python_scripts_dest.exists():
            shutil.rmtree(python_scripts_dest)
        shutil.copytree(python_scripts_src, python_scripts_dest)

    for asset_file in ["logo.png", "icon.png"]:
        asset_src = Path(__file__).parent / asset_file
        asset_dest = INSTALL_DIR / asset_file
        if asset_src.exists():
            shutil.copy2(asset_src, asset_dest)


def create_launcher_script():
    print("🚀 Creating launcher script...")
    content = f"""#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {CONDA_ENV_NAME}
python "{INSTALL_DIR}/main.py"
"""
    with open(LAUNCHER_SCRIPT, "w") as f:
        f.write(content)
    os.chmod(LAUNCHER_SCRIPT, 0o755)
    print(f"✅ Launcher script created at: {LAUNCHER_SCRIPT}")


def create_global_command():
    print("🔗 Creating global terminal command `stablyzegraph`...")
    bin_dir = GLOBAL_LINK_PATH.parent
    bin_dir.mkdir(parents=True, exist_ok=True)
    if GLOBAL_LINK_PATH.exists():
        GLOBAL_LINK_PATH.unlink()
    GLOBAL_LINK_PATH.symlink_to(LAUNCHER_SCRIPT)
    print(f"✅ Global command symlinked at: {GLOBAL_LINK_PATH}")

    shell = os.environ.get("SHELL", "")
    rc_file = "~/.bashrc"
    if "zsh" in shell:
        rc_file = "~/.zshrc"
    rc_path = Path(rc_file).expanduser()
    if rc_path.exists():
        with open(rc_path, "r") as f:
            content = f.read()
        if "export PATH=~/.local/bin:$PATH" not in content:
            with open(rc_path, "a") as f:
                f.write("\n# Added by StablyzeGraph installer\nexport PATH=~/.local/bin:$PATH\n")
            print(f"📌 Added ~/.local/bin to PATH in {rc_file}. Run `source {rc_file}` or restart terminal.")


def create_desktop_entry():
    if platform.system().lower() != "linux":
        return
    print("🖥️ Creating Linux desktop entry...")
    icon_path = INSTALL_DIR / ICON_FILENAME

    entry = f"""[Desktop Entry]
Type=Application
Name=StabLyzeGraph
Exec=bash "{LAUNCHER_SCRIPT}"
Icon={icon_path if icon_path.exists() else ''}
Terminal=false
Categories=Science;
"""
    DESKTOP_ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DESKTOP_ENTRY_PATH, "w") as f:
        f.write(entry)
    os.chmod(DESKTOP_ENTRY_PATH, 0o755)
    print(f"✅ Desktop shortcut created at: {DESKTOP_ENTRY_PATH}")


def create_uninstall_script():
    print("🧹 Creating uninstall script...")
    script = f"""#!/bin/bash
echo "Removing Conda environment '{CONDA_ENV_NAME}'..."
conda remove -y --name {CONDA_ENV_NAME} --all

echo "Removing installation directory..."
rm -rf "{INSTALL_DIR}"

echo "Removing launcher command..."
rm -f "{GLOBAL_LINK_PATH}"

echo "Removing desktop entry..."
rm -f "{DESKTOP_ENTRY_PATH}"

echo "✅ StablyzeGraph uninstalled successfully."
"""
    with open(UNINSTALL_SCRIPT_PATH, "w") as f:
        f.write(script)
    os.chmod(UNINSTALL_SCRIPT_PATH, 0o755)
    print(f"✅ Uninstall script created at: {UNINSTALL_SCRIPT_PATH}")


def main():
    print("=== 🚀 StablyzeGraph Installer ===")

    if not check_command("conda"):
        install_miniconda()
        if not check_command("conda"):
            print("❌ Conda is still not available. Exiting.")
            sys.exit(1)

    create_conda_env()
    install_dependencies()
    install_clustalo()
    copy_application_files()
    create_launcher_script()
    create_global_command()
    create_desktop_entry()
    create_uninstall_script()

    print("\n✅ Installation complete!")
    print("👉 Launch app by typing: `stablyzegraph`")
    print(f"   or run: {LAUNCHER_SCRIPT}")
    print("🧽 To uninstall, run:")
    print(f"   bash {UNINSTALL_SCRIPT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
