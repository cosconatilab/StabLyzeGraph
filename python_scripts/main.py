#!/usr/bin/env python3
"""
StablyzeGraph Desktop Application (Rewritten v5)
A PyQt6-based GUI for protein engineering with Benchmarking and Screening modes.
Focus on simplified process management, enhanced logging, and robustness.
Adds model upload for Screening mode.
Adds Output Directory selection.
Updates layout to match reference images.
Fixes progress bar updates.
Implements automatic results display (plots, metrics, tables).
Adds automatic opening of the output directory on success.
Fixes Screening script arguments.
Updates Benchmarking display to show specific plots (Learning Curve, Probability) and metrics from CSV.
"""

import sys
import os
import json
import subprocess
import threading
import time
import traceback # Added for detailed error logging
from pathlib import Path
from datetime import datetime

import pandas as pd # For reading CSV results
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QFormLayout, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTextEdit, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette, QDesktopServices # For opening URLs/Files
from PyQt6.QtCore import QUrl # For opening URLs/Files

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

# Define color scheme
COLORS = {
    "primary": "#0088a9",  # Blue
    "secondary": "#e8c39e",  # Skin color
    "background": "#f5f5f5",
    "text": "#333333",
    "progress": "#4caf50",  # Green
    "progress_background": "#e0e0e0",
    "button": "#0088a9",
    "button_hover": "#006d87",
    "button_text": "#ffffff",
}

# --- Utility Functions ---

def get_script_path(script_name):
    """Gets the absolute path to a script in the python_scripts directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "python_scripts", script_name)

def get_asset_path(asset_name):
    """Gets the absolute path to an asset (like logo.png)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, asset_name)

def create_default_output_dir(mode):
    """Creates and returns the default output directory path."""
    dir_path = os.path.join(os.path.expanduser("~"), "stablyzegraph_results", mode, datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    except OSError as e:
        print(f"Error creating default output directory: {e}")
        # Fallback to current directory if home directory fails
        fallback_path = os.path.join(os.getcwd(), "stablyzegraph_results", mode, datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            os.makedirs(fallback_path, exist_ok=True)
            return fallback_path
        except OSError as fallback_e:
            print(f"Error creating fallback output directory: {fallback_e}")
            return os.getcwd() # Last resort

def open_directory(path):
    """Opens the specified directory in the default file explorer."""
    if not os.path.isdir(path):
        print(f"Error: Directory not found: {path}")
        return False
    try:
        # Use QDesktopServices for better cross-platform compatibility
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        print(f"Attempted to open directory: {path}")
        return True
    except Exception as e:
        print(f"Error opening directory {path}: {e}")
        # Fallback attempts (less reliable)
        try:
            if platform.system() == "Windows":
                os.startfile(path)
                return True
            elif platform.system() == "Darwin": # macOS
                subprocess.Popen(["open", path])
                return True
            else: # Linux and other Unix-like
                subprocess.Popen(["xdg-open", path])
                return True
        except Exception as fallback_e:
            print(f"Fallback failed to open directory {path}: {fallback_e}")
            return False

# --- Worker Thread --- 

class WorkerThread(QThread):
    """Worker thread for running background processes (Benchmarking/Screening)."""
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    process_finished = pyqtSignal(bool, str, str, str) # success, stdout, stderr, output_dir
    
    def __init__(self, command, working_dir, output_dir):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.output_dir = output_dir # This is now crucial
        self.process = None
        self.is_running = True
        # Ensure output dir exists before defining progress/result files
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            print(f"CRITICAL: Could not create output directory {self.output_dir}: {e}")
            
        self.progress_file = os.path.join(self.output_dir, "progress.json")
        self.result_file = os.path.join(self.output_dir, "results.json")
        
    def run(self):
        stdout_str = ""
        stderr_str = ""
        try:
            self.log_message.emit(f"Worker thread started for command: {' '.join(self.command)}")
            self.log_message.emit(f"Working directory: {self.working_dir}")
            self.log_message.emit(f"Output directory: {self.output_dir}")
            self.log_message.emit(f"Progress file: {self.progress_file}")
            self.log_message.emit(f"Result file: {self.result_file}")
            
            # Clean up old progress/result files in the specific output dir
            try:
                if os.path.exists(self.progress_file): os.remove(self.progress_file)
                if os.path.exists(self.result_file): os.remove(self.result_file)
            except OSError as e:
                self.log_message.emit(f"Warning: Could not remove old status files from {self.output_dir}: {e}")
                
            # Start the process
            # Use conda run to activate the environment and execute the script
            full_command = ["conda", "run", "-n", "stablyzegraph", "python"] + self.command
            self.log_message.emit(f"Executing: {' '.join(full_command)}")
            
            # Define log file paths for subprocess stdout/stderr
            stdout_log_path = os.path.join(self.output_dir, "script_stdout.log")
            stderr_log_path = os.path.join(self.output_dir, "script_stderr.log")
            self.log_message.emit(f"DEBUG: Defined script stdout log path: {stdout_log_path}")
            self.log_message.emit(f"DEBUG: Defined script stderr log path: {stderr_log_path}")

            try:
                self.log_message.emit(f"DEBUG: Attempting to open log files '{stdout_log_path}' and '{stderr_log_path}' for writing.")
                # Open log files for stdout and stderr
                with open(stdout_log_path, "w", encoding="utf-8") as f_stdout, \
                     open(stderr_log_path, "w", encoding="utf-8") as f_stderr:
                    self.log_message.emit(f"DEBUG: Log files opened successfully. Preparing to execute Popen.")
                    self.log_message.emit(f"DEBUG: Popen command: {' '.join(full_command)}")
                    self.log_message.emit(f"DEBUG: Popen CWD: {self.working_dir}")
                    
                    self.process = subprocess.Popen(
                        full_command,
                        cwd=self.working_dir,
                        stdout=f_stdout, 
                        stderr=f_stderr, 
                        text=True, 
                        bufsize=1, 
                        encoding="utf-8",
                        errors="replace"
                    )
                    # If Popen succeeds, this line will be reached.
                    self.log_message.emit(f"DEBUG: subprocess.Popen executed. PID: {self.process.pid if self.process else 'N/A - Popen might have failed silently if process is None'}")

            except FileNotFoundError as fnf_error:
                error_details = traceback.format_exc()
                self.log_message.emit(f"CRITICAL ERROR: FileNotFoundError during Popen setup or execution (e.g., script '{full_command[0]}' not found, or CWD '{self.working_dir}' invalid): {fnf_error}\nTraceback:\n{error_details}")
                self.process_finished.emit(False, "", f"Failed to start script (FileNotFound): {fnf_error}. Check script/Python path and CWD.", self.output_dir)
                return # Exit the run method
            except PermissionError as perm_error:
                error_details = traceback.format_exc()
                self.log_message.emit(f"CRITICAL ERROR: PermissionError during Popen setup or execution (cannot access script/interpreter, CWD, or open log files): {perm_error}\nTraceback:\n{error_details}")
                self.process_finished.emit(False, "", f"Failed to start script (PermissionError): {perm_error}", self.output_dir)
                return # Exit the run method
            except OSError as os_error: # Catch other OS-level errors, e.g., invalid argument for Popen
                error_details = traceback.format_exc()
                self.log_message.emit(f"CRITICAL ERROR: OSError during Popen setup or execution: {os_error}\nTraceback:\n{error_details}")
                self.process_finished.emit(False, "", f"Failed to start script (OSError): {os_error}", self.output_dir)
                return # Exit the run method
            except Exception as e:
                # Catch any other unexpected errors during Popen or log file handling
                error_details = traceback.format_exc()
                self.log_message.emit(f"CRITICAL UNEXPECTED ERROR during Popen setup or execution: {e}\nTraceback:\n{error_details}")
                self.process_finished.emit(False, "", f"Unexpected error starting script: {e}", self.output_dir)
                return # Exit the run method

            # Check if self.process was successfully created
            if not self.process or not hasattr(self.process, 'pid'):
                self.log_message.emit(f"CRITICAL ERROR: self.process object was not created or is invalid after Popen block, but no specific exception was caught. This indicates a serious issue with Popen.")
                self.process_finished.emit(False, "", "Internal error: Failed to initialize script process object.", self.output_dir)
                return

            # This log message confirms self.process is valid and has a PID
            self.log_message.emit(f"Process started successfully (PID: {self.process.pid})")
            
            # Monitor progress file and stdout/stderr
            last_progress_check_time = time.time()
            while self.is_running and self.process.poll() is None:
                current_time = time.time()
                # Check progress file ~every 0.5 seconds
                if current_time - last_progress_check_time > 0.5:
                    if os.path.exists(self.progress_file):
                        try:
                            with open(self.progress_file, "r") as f:
                                progress_data = json.load(f)
                            progress = progress_data.get("progress", 0)
                            message = progress_data.get("message", "Running...")
                            self.progress_updated.emit(progress, message)
                        except (json.JSONDecodeError, FileNotFoundError, PermissionError, ValueError) as e:
                            pass # Ignore transient errors
                    else:
                        if time.time() - last_progress_check_time < 2:
                             self.progress_updated.emit(0, "Initializing...")
                    last_progress_check_time = current_time
                
                time.sleep(0.1) # Small sleep to yield CPU

            if not self.is_running:
                 self.log_message.emit("Process cancelled by user.")
                 self.process_finished.emit(False, "", "Process cancelled by user.", self.output_dir)
                 return
                 
            self.log_message.emit("Process finished, collecting results...")
            stdout, stderr = self.process.communicate()
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            return_code = self.process.returncode
            self.log_message.emit(f"Process exited with code: {return_code}")
            if stdout: self.log_message.emit(f"Process STDOUT:\n{stdout}")
            if stderr: self.log_message.emit(f"Process STDERR:\n{stderr}")
            
            result_data = {}
            success = False
            final_error_message = stderr if stderr is not None else ""
            
            # Check results.json first for definitive status
            if os.path.exists(self.result_file):
                self.log_message.emit(f"Results file found: {self.result_file}")
                try:
                    with open(self.result_file, "r") as f:
                        result_data = json.load(f)
                    success = result_data.get("status") == "success"
                    if not success:
                        json_error = result_data.get("error")
                        if json_error: 
                            final_error_message = json_error
                            self.log_message.emit(f"Error reported in results file: {json_error}")
                        else:
                            self.log_message.emit("Results file indicates failure but contains no specific error message.")
                            if not final_error_message:
                                final_error_message = "Process failed. Check logs for details."
                    else:
                         self.log_message.emit("Results file indicates success.")
                         final_error_message = "" # Clear any previous stderr if success
                except (json.JSONDecodeError, PermissionError, ValueError) as e:
                    final_error_message += f"\nError reading results file ({self.result_file}): {e}"
                    self.log_message.emit(f"Error reading results file: {e}")
                    success = False
            elif return_code == 0:
                 final_error_message += "\nError: Process finished successfully (exit code 0) but no results.json file was found."
                 self.log_message.emit(final_error_message)
                 success = False
            elif not final_error_message:
                final_error_message = f"Process exited with code {return_code} but produced no error message."
                if stdout: final_error_message += f"\nSTDOUT:\n{stdout}"
                self.log_message.emit(final_error_message)
                success = False
            else:
                 self.log_message.emit(f"Process failed with stderr, no results file found (as expected). Error: {final_error_message}")
                 success = False
            
            # Emit final status
            self.process_finished.emit(success, stdout, final_error_message, self.output_dir)
            self.log_message.emit(f"Worker thread finished. Success: {success}")
            
        except Exception as e:
            error_msg = f"An unexpected error occurred in the worker thread: {e}"
            self.log_message.emit(error_msg)
            stderr_str = f"{stderr}\n{error_msg}"
            self.process_finished.emit(False, stdout, stderr_str, self.output_dir)

    def stop(self):
        self.log_message.emit("Stopping worker thread...")
        self.is_running = False
        if self.process and self.process.poll() is None:
            try:
                self.log_message.emit(f"Terminating process {self.process.pid}...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                    self.log_message.emit("Process terminated gracefully.")
                except subprocess.TimeoutExpired:
                    self.log_message.emit("Process did not terminate gracefully, killing...")
                    self.process.kill()
                    self.log_message.emit("Process killed.")
            except Exception as e:
                self.log_message.emit(f"Error during process termination: {e}")
        else:
             self.log_message.emit("Process already finished or not started.")

# --- Matplotlib Canvas --- 

class MatplotlibCanvas(FigureCanvas):
    """Canvas for displaying matplotlib plots."""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
        self.clear_plot()

    def display_plot(self, plot_file):
        self.axes.clear()
        if plot_file and os.path.exists(plot_file):
            try:
                img = mpimg.imread(plot_file)
                self.axes.imshow(img)
                self.axes.axis("off")
            except Exception as e:
                print(f"Error loading plot {plot_file}: {e}")
                self.axes.text(0.5, 0.5, f"Error loading plot:\n{os.path.basename(plot_file)}", 
                                 ha="center", va="center", color="red")
        else:
            self.axes.text(0.5, 0.5, "Plot not available", ha="center", va="center")
        try:
            self.draw()
        except Exception as e:
             print(f"Error drawing plot canvas: {e}")
        
    def clear_plot(self):
        self.axes.clear()
        self.axes.text(0.5, 0.5, "Results will appear here", ha="center", va="center")
        try:
            self.draw()
        except Exception as e:
             print(f"Error drawing cleared plot canvas: {e}")

# --- Base Tab Class --- 

class BaseTab(QWidget):
    """Base class for Benchmarking and Screening tabs."""
    log_message_signal = pyqtSignal(str)
    
    def __init__(self, mode_name, script_name, parent=None):
        super().__init__(parent)
        self.mode_name = mode_name
        self.script_name = script_name
        self.parent_window = parent
        self.worker_thread = None
        self.file_inputs = {}
        self.param_inputs = {}
        self.output_dir_edit = None
        self.current_output_dir = None
        self.setup_ui()
        if self.parent_window:
            self.log_message_signal.connect(self.parent_window.log_message)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === Parameters Section (Left Side) ===
        params_widget = QScrollArea()
        params_widget.setWidgetResizable(True)
        params_content = QWidget()
        self.params_layout = QVBoxLayout(params_content)
        params_widget.setWidget(params_content)
        params_widget.setMinimumWidth(350)
        params_widget.setMaximumWidth(450)

        self._create_file_inputs_group()
        self._create_model_params_group()
        self._create_output_group()
        self._create_run_button()
        self._create_progress_group()
        self.params_layout.addStretch()

        # === Results Section (Right Side) ===
        results_widget = QWidget()
        self.results_layout = QVBoxLayout(results_widget)
        self._create_results_display() # Subclasses implement this

        # Add widgets to splitter and main layout
        splitter.addWidget(params_widget)
        splitter.addWidget(results_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _create_file_inputs_group(self):
        self.file_input_group = QGroupBox("Input Files")
        self.file_input_layout = QFormLayout()
        self.file_input_group.setLayout(self.file_input_layout)
        self.params_layout.addWidget(self.file_input_group)

    def _add_file_input(self, key, label, file_filter):
        edit = QLineEdit()
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        row_layout = QHBoxLayout()
        row_layout.addWidget(edit)
        row_layout.addWidget(btn)
        self.file_input_layout.addRow(label, row_layout)
        btn.clicked.connect(lambda checked=False, le=edit, ff=file_filter: self._browse_file(le, ff))
        self.file_inputs[key] = edit
        return edit

    def _create_model_params_group(self):
        self.model_params_group = QGroupBox("Model Parameters")
        self.model_params_layout = QFormLayout()
        self.model_params_group.setLayout(self.model_params_layout)
        self.params_layout.addWidget(self.model_params_group)

    def _add_param_input(self, key, label, widget_type, default_value, **kwargs):
        if widget_type == "spinbox":
            widget = QSpinBox()
            if "min" in kwargs: widget.setMinimum(kwargs["min"])
            if "max" in kwargs: widget.setMaximum(kwargs["max"])
            if "step" in kwargs: widget.setSingleStep(kwargs["step"])
            widget.setValue(default_value)
        elif widget_type == "doublespinbox":
            widget = QDoubleSpinBox()
            if "min" in kwargs: widget.setMinimum(kwargs["min"])
            if "max" in kwargs: widget.setMaximum(kwargs["max"])
            if "step" in kwargs: widget.setSingleStep(kwargs["step"])
            if "decimals" in kwargs: widget.setDecimals(kwargs["decimals"])
            widget.setValue(default_value)
        elif widget_type == "combobox":
            widget = QComboBox()
            if "items" in kwargs: widget.addItems(kwargs["items"])
            widget.setCurrentText(default_value)
        else:
            widget = QLineEdit()
            widget.setText(str(default_value))
            
        self.model_params_layout.addRow(label, widget)
        self.param_inputs[key] = widget
        return widget

    def _create_output_group(self):
        self.output_group = QGroupBox("Output")
        self.output_layout = QFormLayout()
        self.output_group.setLayout(self.output_layout)
        self.params_layout.addWidget(self.output_group)

        self.output_dir_edit = QLineEdit()
        default_dir = create_default_output_dir(self.mode_name.lower())
        self.output_dir_edit.setText(default_dir)
        
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        row_layout = QHBoxLayout()
        row_layout.addWidget(self.output_dir_edit)
        row_layout.addWidget(btn)
        self.output_layout.addRow("Output Directory:", row_layout)
        btn.clicked.connect(self._browse_directory)

    def _create_run_button(self):
        self.run_button = QPushButton(f"Run {self.mode_name}")
        self.run_button.setStyleSheet(f"background-color: {COLORS['button']}; color: {COLORS['button_text']}; padding: 8px;")
        self.run_button.clicked.connect(self._run_process)
        self.params_layout.addWidget(self.run_button)

    def _create_progress_group(self):
        self.progress_group = QGroupBox("Progress")
        self.progress_layout = QVBoxLayout()
        self.progress_group.setLayout(self.progress_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - Ready")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: {COLORS["progress_background"]};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS["progress"]};
                width: 20px;
            }}
        """)
        self.progress_layout.addWidget(self.progress_bar)
        self.params_layout.addWidget(self.progress_group)

    def _browse_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {line_edit.objectName()}", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def _browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _run_process(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self._stop_process()
            return

        # --- Prepare Command ---
        script_path = get_script_path(self.script_name)
        command = [script_path]

        # Add file inputs to the command
        for key, edit in self.file_inputs.items():
            command.extend([f"--{key}", edit.text()])

        # Add parameter inputs to the command
        for key, widget in self.param_inputs.items():
            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            else:
                value = widget.text()
            command.extend([f"--{key}", str(value)])

        # # Add optimal_threshold to the command
        # optimal_threshold_value = self.param_inputs["optimal_threshold"].value()
        # command.extend(["--optimal_threshold", str(optimal_threshold_value)])

        # Validate Inputs
        missing_files = []
        for key, edit in self.file_inputs.items():
            if not edit.text() or not os.path.exists(edit.text()):
                missing_files.append(key)

        if not self.output_dir_edit.text():
            missing_files.append("Output Directory")

        if missing_files:
            QMessageBox.warning(self, "Missing Inputs", f"Please provide valid paths for: {', '.join(missing_files)}")
            return

        # Set up output directory
        self.current_output_dir = self.output_dir_edit.text()
        try:
            os.makedirs(self.current_output_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not create output directory:\n{self.current_output_dir}\nError: {e}")
            return

        command.extend(["--output", self.current_output_dir])
        progress_file_path = os.path.join(self.current_output_dir, "progress.json")
        command.extend(["--progress_file", progress_file_path])

        # --- Clear Previous Results ---
        self.clear_results_display()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% - Starting...")
        if self.parent_window:
            self.parent_window.log_output.clear()
        self.log_message_signal.emit(f"Starting {self.mode_name}...")
        self.log_message_signal.emit(f"Output Directory: {self.current_output_dir}")

        # --- Start Worker ---
        self.worker_thread = WorkerThread(command, os.path.dirname(script_path), self.current_output_dir)
        self.worker_thread.progress_updated.connect(self._update_progress)
        self.worker_thread.log_message.connect(self.log_message_signal.emit)
        self.worker_thread.process_finished.connect(self._on_process_finished)
        self.worker_thread.start()

        self.run_button.setText(f"Stop {self.mode_name}")
        self.run_button.setStyleSheet(f"background-color: red; color: white; padding: 8px;")


    def _stop_process(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message_signal.emit(f"Attempting to stop {self.mode_name}...")
            self.worker_thread.stop()

    def _update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")

    def _on_process_finished(self, success, stdout, stderr, output_dir):
        self.log_message_signal.emit(f"{self.mode_name} finished. Success: {success}")
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100% - Complete")
            QMessageBox.information(self, "Success", f"{self.mode_name} completed successfully!")
            self.display_results(output_dir)
            if not open_directory(output_dir):
                 QMessageBox.warning(self, "Warning", f"Could not automatically open output directory:\n{output_dir}\nPlease navigate to it manually.")
        else:
            self.progress_bar.setFormat(f"Error - {stderr.splitlines()[-1] if stderr else 'Unknown Error'}")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    background-color: {COLORS["progress_background"]};
                }}
                QProgressBar::chunk {{
                    background-color: red;
                    width: 20px;
                }}
            """)
            QMessageBox.critical(self, "Error", f"{self.mode_name} failed:\n{stderr}")

        # Reset button and worker thread reference
        self.run_button.setText(f"Run {self.mode_name}")
        self.run_button.setStyleSheet(f"background-color: {COLORS['button']}; color: {COLORS['button_text']}; padding: 8px;")
        self.worker_thread = None
        self.current_output_dir = None

    def _create_results_display(self):
        # To be implemented by subclasses
        pass

    def display_results(self, output_dir):
        # To be implemented by subclasses
        self.log_message_signal.emit(f"Displaying results from: {output_dir}")
        pass
        
    def clear_results_display(self):
        # To be implemented by subclasses
        pass

# --- Benchmarking Tab --- 

class BenchmarkingTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__("Benchmarking", "Benchmarking.py", parent)

    def _create_file_inputs_group(self):
        super()._create_file_inputs_group()
        self._add_file_input("active", "Active Sequences (.csv):", "CSV Files (*.csv)")
        self._add_file_input("inactive", "Inactive Sequences (.csv):", "CSV Files (*.csv)")
        self._add_file_input("wild_type", "Wild Type Sequence (.fasta):", "FASTA Files (*.fasta *.fa)")
        self._add_file_input("pdb", "PDB Structure (.pdb):", "PDB Files (*.pdb)")
        self._add_file_input("properties", "AA Properties (.csv):", "CSV Files (*.csv)")

    def _create_model_params_group(self):
        super()._create_model_params_group()
        self._add_param_input("hidden_dim", "Hidden Dimensions:", "spinbox", 128, min=16, max=1024, step=16)
        self._add_param_input("dropout_rate", "Dropout Rate:", "doublespinbox", 0.25, min=0.0, max=1.0, step=0.05, decimals=2)
        self._add_param_input("ratio", "Pooling Ratio:", "doublespinbox", 0.70, min=0.1, max=1.0, step=0.05, decimals=2)
        self._add_param_input("learning_rate", "Learning Rate:", "doublespinbox", 0.00001, min=1e-7, max=1e-2, step=1e-6, decimals=7)
        self._add_param_input("l2_regularization", "L2 Regularization:", "doublespinbox", 0.0001, min=0.0, max=0.1, step=1e-5, decimals=5)
        self._add_param_input("scheduler_factor", "Scheduler Factor:", "doublespinbox", 0.9, min=0.1, max=0.99, step=0.05, decimals=2)
        self._add_param_input("scheduler_patience", "Scheduler Patience:", "spinbox", 10, min=1, max=100)
        self._add_param_input("stop_patience", "Early Stop Patience:", "spinbox", 50, min=5, max=500)
        self._add_param_input("grad_clip", "Gradient Clipping:", "doublespinbox", 10.0, min=0.1, max=100.0, step=0.5, decimals=1)
        self._add_param_input("max_epochs", "Max Epochs:", "spinbox", 1000, min=10, max=10000)
        self._add_param_input("seed", "Random Seed:", "spinbox", 42, min=0, max=99999)
        self._add_param_input("num_cores", "CPU Cores (-1 for all):", "spinbox", -1, min=-1, max=os.cpu_count() or 1)

    def _create_results_display(self):
        results_group = QGroupBox("Benchmarking Results")
        results_group_layout = QVBoxLayout()
        results_group.setLayout(results_group_layout)
        
        # Splitter for plots and metrics table
        results_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top part: Plots (Learning Curve and Probability Plots)
        plots_group = QGroupBox("Plots")
        plots_layout = QHBoxLayout() # Use horizontal layout for two plots side-by-side
        plots_group.setLayout(plots_layout)

        self.learning_curve_canvas = MatplotlibCanvas(width=5, height=4)
        self.probability_plots_canvas = MatplotlibCanvas(width=5, height=4)
        
        plots_layout.addWidget(self.learning_curve_canvas)
        plots_layout.addWidget(self.probability_plots_canvas)
        
        results_splitter.addWidget(plots_group)

        # Bottom part: Metrics Table
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False) # Hide row numbers
        metrics_layout.addWidget(self.metrics_table)
        results_splitter.addWidget(metrics_group)
        
        results_splitter.setSizes([500, 300]) # Adjust initial sizes
        results_group_layout.addWidget(results_splitter)
        self.results_layout.addWidget(results_group)
        self.clear_results_display() # Show empty state initially

    def display_results(self, output_dir):
        super().display_results(output_dir)
        results_file = os.path.join(output_dir, "results.json")
        metrics_csv_file = os.path.join(output_dir, "metrics.csv") # Default name
        
        plot_paths = {}
        
        # Read results.json to get plot paths and metrics file path
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                if results_data.get("status") == "success":
                    plot_paths = results_data.get("plots", {})
                    # Get metrics file path from results.json if available
                    metrics_file_from_results = results_data.get("metrics_file")
                    if metrics_file_from_results and os.path.exists(metrics_file_from_results):
                         metrics_csv_file = metrics_file_from_results
            except Exception as e:
                self.log_message_signal.emit(f"Error reading results file {results_file}: {e}")
        else:
             self.log_message_signal.emit(f"Warning: results.json not found in {output_dir}")
             # Try default plot names if results.json is missing
             plot_paths = {
                 "learning_curve": os.path.join(output_dir, "learning_curve_plot.png"),
                 "probability_plots": os.path.join(output_dir, "probability_plots.png"),
             }

        # Display plots
        self.learning_curve_canvas.display_plot(plot_paths.get("learning_curve"))
        self.probability_plots_canvas.display_plot(plot_paths.get("probability_plots"))

        # Display metrics from CSV in the table
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(0)
        if os.path.exists(metrics_csv_file):
            try:
                df = pd.read_csv(metrics_csv_file)
                # Assuming metrics CSV has metrics as columns and one row of values
                headers = df.columns.tolist()
                self.metrics_table.setColumnCount(len(headers))
                self.metrics_table.setHorizontalHeaderLabels(headers)
                self.metrics_table.setRowCount(len(df))
                
                for i, row in df.iterrows():
                    for j, col_name in enumerate(headers):
                        value = row[col_name]
                        # Format floating point numbers nicely
                        if isinstance(value, float):
                            item = QTableWidgetItem(f"{value:.4f}")
                        else:
                            item = QTableWidgetItem(str(value))
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.metrics_table.setItem(i, j, item)
                
                self.metrics_table.resizeColumnsToContents()
                self.log_message_signal.emit(f"Displayed metrics from {metrics_csv_file}")
            except Exception as e:
                self.log_message_signal.emit(f"Error reading or displaying metrics CSV {metrics_csv_file}: {e}")
                # Show error in table
                self.metrics_table.setColumnCount(1)
                self.metrics_table.setRowCount(1)
                self.metrics_table.setHorizontalHeaderLabels(["Error"])
                error_item = QTableWidgetItem(f"Error loading metrics: {e}")
                self.metrics_table.setItem(0, 0, error_item)
        else:
            self.log_message_signal.emit(f"Warning: Metrics file not found: {metrics_csv_file}")
            # Show placeholder in table
            self.metrics_table.setColumnCount(1)
            self.metrics_table.setRowCount(1)
            self.metrics_table.setHorizontalHeaderLabels(["Status"])
            placeholder_item = QTableWidgetItem("Metrics file not found.")
            self.metrics_table.setItem(0, 0, placeholder_item)
            
    def clear_results_display(self):
        self.learning_curve_canvas.clear_plot()
        self.probability_plots_canvas.clear_plot()
        # Clear metrics table
        self.metrics_table.setRowCount(0)
        self.metrics_table.setColumnCount(1)
        self.metrics_table.setHorizontalHeaderLabels(["Status"])
        self.metrics_table.setRowCount(1)
        placeholder_item = QTableWidgetItem("Metrics will appear here after successful run.")
        placeholder_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_table.setItem(0, 0, placeholder_item)

# --- Screening Tab --- 

class ScreeningTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__("Screening", "Screening.py", parent)

    def _create_file_inputs_group(self):
        super()._create_file_inputs_group()
        self._add_file_input("active", "Active Sequences (.csv):", "CSV Files (*.csv)")
        self._add_file_input("inactive", "Inactive Sequences (.csv):", "CSV Files (*.csv)")
        self._add_file_input("wild_type", "Wild Type Sequence (.fasta):", "FASTA Files (*.fasta *.fa)")
        self._add_file_input("pdb", "PDB Structure (.pdb):", "PDB Files (*.pdb)")
        self._add_file_input("properties", "AA Properties (.csv):", "CSV Files (*.csv)")
        self._add_file_input("model_path", "Trained Model (.pth):", "PyTorch Model Files (*.pth)")

    def _create_model_params_group(self):
        super()._create_model_params_group()
        self._add_param_input("hidden_dim", "Hidden Dimensions (match model):", "spinbox", 128, min=16, max=1024, step=16)
        self._add_param_input("dropout_rate", "Dropout Rate (match model):", "doublespinbox", 0.25, min=0.0, max=1.0, step=0.05, decimals=2)
        self._add_param_input("ratio", "Pooling Ratio (match model):", "doublespinbox", 0.70, min=0.1, max=1.0, step=0.05, decimals=2)
        self._add_param_input("num_mutations", "Number of Mutations:", "spinbox", 2, min=1, max=10)
        self._add_param_input("generation_method", "Mutant Generation Method:", "combobox", "combinatorial", items=["combinatorial", "weighted", "evolutionary"])
        self._add_param_input("max_mutants", "Max Mutants to Screen:", "spinbox", 5000, min=10, max=100000, step=100)
        self._add_param_input("seed", "Random Seed:", "spinbox", 42, min=0, max=99999)
        # self._add_param_input("optimal_threshold", "Optimal Threshold:", "doublespinbox", 0.5, min=0.0, max=1.0, step=0.01, decimals=3)  # Add this line
        self._add_param_input("num_cores", "CPU Cores (-1 for all):", "spinbox", -1, min=-1, max=os.cpu_count() or 1)

    def _create_results_display(self):
        results_group = QGroupBox("Screening Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Rank", "Mutation", "Predicted Probability"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        
        self.results_layout.addWidget(results_group)
        self.clear_results_display()

    def display_results(self, output_dir):
        super().display_results(output_dir)
        results_file = os.path.join(output_dir, "results.json")
        ranked_mutants_file = os.path.join(output_dir, "ranked_mutants.csv")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                if results_data.get("status") == "success":
                    file_from_results = results_data.get("ranked_mutants_file")
                    if file_from_results and os.path.exists(file_from_results):
                        ranked_mutants_file = file_from_results
            except Exception as e:
                self.log_message_signal.emit(f"Error reading results file {results_file}: {e}")
        
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(0)
        if os.path.exists(ranked_mutants_file):
            try:
                df = pd.read_csv(ranked_mutants_file)
                df_display = df.head(20) # Limit to top 20
                self.results_table.setRowCount(len(df_display))
                for i, row in df_display.iterrows():
                    rank_item = QTableWidgetItem(str(row.get("rank", i + 1)))
                    mutation_item = QTableWidgetItem(str(row.get("mutation", "N/A")))
                    prob_item = QTableWidgetItem(f"{row.get('probability', 0.0):.4f}")
                    
                    rank_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    prob_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    
                    self.results_table.setItem(i, 0, rank_item)
                    self.results_table.setItem(i, 1, mutation_item)
                    self.results_table.setItem(i, 2, prob_item)
                self.log_message_signal.emit(f"Displayed {len(df)} ranked mutants from {ranked_mutants_file}")
            except Exception as e:
                self.log_message_signal.emit(f"Error reading or displaying ranked mutants file {ranked_mutants_file}: {e}")
                self.results_table.setRowCount(1)
                error_item = QTableWidgetItem(f"Error loading results: {e}")
                error_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setSpan(0, 0, 1, 3)
                self.results_table.setItem(0, 0, error_item)
        else:
            self.log_message_signal.emit(f"Warning: Ranked mutants file not found: {ranked_mutants_file}")
            self.results_table.setRowCount(1)
            no_results_item = QTableWidgetItem("Ranked mutants file not found.")
            no_results_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setSpan(0, 0, 1, 3)
            self.results_table.setItem(0, 0, no_results_item)
            
        self.results_table.setSortingEnabled(True)
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def clear_results_display(self):
        self.results_table.setRowCount(0)
        self.results_table.setRowCount(1)
        placeholder_item = QTableWidgetItem("Results will appear here after successful run.")
        placeholder_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_table.setSpan(0, 0, 1, 3)
        self.results_table.setItem(0, 0, placeholder_item)

# --- Main Window --- 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StablyzeGraph")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['background']}; color: {COLORS['text']};")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_path = get_asset_path("logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            logo_label.setText("[Logo]")
        title_label = QLabel("StablyzeGraph")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['primary']}; margin-left: 10px;")
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Content Splitter (Tabs and Log)
        content_splitter = QSplitter(Qt.Orientation.Vertical)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { padding: 10px; }
            QTabWidget::pane { border-top: 1px solid #ccc; }
        """)
        self.benchmarking_tab = BenchmarkingTab(self)
        self.screening_tab = ScreeningTab(self)
        self.tabs.addTab(self.benchmarking_tab, "Benchmarking")
        self.tabs.addTab(self.screening_tab, "Screening")
        content_splitter.addWidget(self.tabs)

        # Log Output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("monospace")
        self.log_output.setStyleSheet("background-color: #eee;")
        log_layout.addWidget(self.log_output)
        content_splitter.addWidget(log_group)
        
        content_splitter.setSizes([600, 200])
        main_layout.addWidget(content_splitter)

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def closeEvent(self, event):
        stopped_bench = True
        if self.benchmarking_tab.worker_thread and self.benchmarking_tab.worker_thread.isRunning():
            reply = QMessageBox.question(self, "Confirm Exit", 
                                       "Benchmarking is still running. Stop and exit?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.benchmarking_tab._stop_process()
                self.benchmarking_tab.worker_thread.wait(1000) 
            else:
                stopped_bench = False
                event.ignore()

        stopped_screen = True
        if self.screening_tab.worker_thread and self.screening_tab.worker_thread.isRunning():
            reply = QMessageBox.question(self, "Confirm Exit", 
                                       "Screening is still running. Stop and exit?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                       QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.screening_tab._stop_process()
                self.screening_tab.worker_thread.wait(1000)
            else:
                stopped_screen = False
                event.ignore()
                
        if stopped_bench and stopped_screen:
            event.accept()
        else:
             event.ignore()

# --- Main Execution --- 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if "Fusion" in QApplication.style().objectName():
        QApplication.setStyle("Fusion")
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

# --- End of File ---               