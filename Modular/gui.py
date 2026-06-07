import os
import sys
import re
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QPushButton, QTextEdit, QLabel,
    QSplitter, QListWidget, QFileDialog, QMessageBox, QGroupBox,
    QInputDialog, QComboBox, QCheckBox, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QProcess, QSize, QEvent, QPointF
from PySide6.QtGui import QPixmap, QIcon


RUNNER_DISPATCH = {
    "run_expanding_n.py": ("run_expanding_n", "main"),
    "run_chronological_chunks.py": ("run_chronological_chunks", "main"),
    "run_trigger_stats.py": ("run_trigger_stats", "main"),
    "run_psth.py": ("run_psth", "main"),
    "run_first_last_block.py": ("run_first_last_block", "main"),
    "run_psth_matrix.py": ("run_psth_matrix", "main"),
    "run_transition_psth.py": ("run_transition_psth", "main"),
    "run_session_diagnostics.py": ("run_session_diagnostics", "main"),
}


def execute_runner(script_name):
    """Executes a runner script by module name, used by frozen and source modes."""
    if script_name not in RUNNER_DISPATCH:
        print(f"Unknown runner: {script_name}")
        return 1

    module_name, func_name = RUNNER_DISPATCH[script_name]
    module = __import__(module_name, fromlist=[func_name])
    getattr(module, func_name)()
    return 0

def natural_sort_key(s):
    """Helper to sort strings containing numbers in a human-friendly way."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class FerretPipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ferret Neural Pipeline Manager")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ferret256.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
        self.resize(1200, 800)
        self.current_pixmap = None
        self.image_zoom = 1.0
        self.image_zoom_step = 1.15
        self.image_zoom_min = 0.2
        self.image_zoom_max = 8.0
        self.is_panning = False
        self.pan_last_global_pos = QPointF()

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # 1. Left Panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.left_panel_content = QWidget()
        left_content_layout = QVBoxLayout(self.left_panel_content)
        self.setup_data_selection_group(left_content_layout)
        self.setup_parameters_group(left_content_layout)
        self.setup_actions_group(left_content_layout)

        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        left_content_layout.addWidget(QLabel("<b>Console Output:</b>"))
        left_content_layout.addWidget(self.console)

        control_layout.addWidget(self.left_panel_content)

        # 2. Right Panel: Results Viewer
        results_panel = QSplitter(Qt.Horizontal)
        
        # 2a. Left Side of Results Viewer: Folder Dropdown + File List
        results_list_container = QWidget()
        results_list_layout = QVBoxLayout(results_list_container)
        results_list_layout.setContentsMargins(0, 0, 0, 0)
        
        self.folder_combo = QComboBox()
        self.folder_combo.currentIndexChanged.connect(self.refresh_file_list)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_image)
        self.file_list.currentItemChanged.connect(self.handle_current_file_changed)
        
        results_list_layout.addWidget(QLabel("<b>Select Run Folder:</b>"))
        results_list_layout.addWidget(self.folder_combo)
        results_list_layout.addWidget(self.file_list)
        
        # 2b. Right Side of Results Viewer: Image Display
        self.viewer_stack = QStackedWidget()

        # Page 0: Image
        self.image_label = QLabel("Select a file to view")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        
        # Page 1: CSV Table
        self.csv_table = QTableWidget()
        self.csv_table.setEditTriggers(QTableWidget.NoEditTriggers) # Read-only
        self.csv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        
        # Page 2: Text Viewer
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        
        self.viewer_stack.addWidget(self.image_scroll) # Index 0
        self.viewer_stack.addWidget(self.csv_table)   # Index 1
        self.viewer_stack.addWidget(self.text_viewer) # Index 2

        results_panel.addWidget(results_list_container)
        results_panel.addWidget(self.viewer_stack)
        results_panel.setSizes([250, 600])

        self.main_splitter.addWidget(control_panel)
        self.main_splitter.addWidget(results_panel)
        self.main_splitter.setSizes([450, 750])

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        # Populate the folders on startup
        self.refresh_run_folders()

    def setup_data_selection_group(self, parent_layout):
        group_box = QGroupBox("Data Selection")
        layout = QVBoxLayout(group_box)

        self.btn_toggle_data_selection = QPushButton("Hide Data Selection")
        self.btn_toggle_data_selection.setCheckable(True)
        self.btn_toggle_data_selection.setChecked(True)
        self.btn_toggle_data_selection.clicked.connect(self.toggle_data_selection)
        layout.addWidget(self.btn_toggle_data_selection)

        self.data_selection_container = QWidget()
        data_layout = QVBoxLayout(self.data_selection_container)
        
        # Default dir setup
        self.default_data_dir = QLineEdit(r"C:\Users\PenPen\Desktop\Ferret\Data")
        data_layout.addWidget(QLabel("Default Starting Folder:"))
        data_layout.addWidget(self.default_data_dir)

        # Action Buttons for Data
        btn_layout = QHBoxLayout()
        self.btn_scan_folder = QPushButton("Scan Parent Folder")
        self.btn_select_files = QPushButton("Select Manual Files")
        self.btn_scan_folder.clicked.connect(self.scan_for_sessions)
        self.btn_select_files.clicked.connect(self.select_manual_files)
        btn_layout.addWidget(self.btn_scan_folder)
        btn_layout.addWidget(self.btn_select_files)
        data_layout.addLayout(btn_layout)
        
        # --- MOVED HERE: Delete Cache Button ---
        self.btn_clear_cache = QPushButton("Delete Cache (Force Re-process)")
        self.btn_clear_cache.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        self.btn_clear_cache.clicked.connect(self.delete_cache)
        data_layout.addWidget(self.btn_clear_cache)
        # ---------------------------------------

        self.session_list_widget = QListWidget()
        data_layout.addWidget(QLabel("<b>Selected Data:</b>"))
        data_layout.addWidget(self.session_list_widget)
        
        self.btn_clear_sessions = QPushButton("Clear Selected Data List")
        self.btn_clear_sessions.clicked.connect(self.session_list_widget.clear)
        data_layout.addWidget(self.btn_clear_sessions)

        layout.addWidget(self.data_selection_container)
        
        parent_layout.addWidget(group_box)

    def toggle_data_selection(self):
        """Toggles the visibility of the data selection container."""
        is_checked = self.btn_toggle_data_selection.isChecked()
        self.data_selection_container.setVisible(is_checked)

        if is_checked:
            self.btn_toggle_data_selection.setText("Hide Data Selection")
        else:
            self.btn_toggle_data_selection.setText("Show Data Selection")

    def scan_for_sessions(self):
        # Fetch the path from the new text box
        start_dir = self.default_data_dir.text() 
        
        # Pass start_dir into the dialog
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory to Scan", start_dir)
        if not parent_dir:
            return
            
        dt = self.dt_input.text().strip()
        target_file = f"data_{dt}.npy"
        found = 0
        
        self.console.append(f"Scanning {parent_dir} for '{target_file}'...")
        QApplication.processEvents()
        
        for root, dirs, files in os.walk(parent_dir):
            if target_file in files:
                item_text = f"SCANNED | {root}"
                self.add_unique_session_item(item_text)
                found += 1
                
        if found > 0:
            self.console.append(f"<font color='green'>Found {found} valid session folders!</font>")
        else:
            QMessageBox.information(self, "No Sessions Found", f"Could not find any folders containing {target_file}.")

    def select_manual_files(self):
        # Fetch the path from the new text box
        start_dir = self.default_data_dir.text()
        
        # Pass start_dir into the dialog (replacing the empty string "")
        files, _ = QFileDialog.getOpenFileNames(self, "Select Manual Data Files", start_dir, "Numpy Files (*.npy);;All Files (*)")
        if not files:
            return
            
        default_dt = float(self.dt_input.text().strip())
        dt_val, ok = QInputDialog.getDouble(
            self, "Custom DT", 
            "Enter the dt value for the manually selected file(s)\n(Used to find the associated features file):", 
            default_dt, 0.0001, 1.0, 4
        )
        
        if not ok:
            return 
            
        for f in files:
            item_text = f"MANUAL | dt={dt_val} | {f}"
            self.add_unique_session_item(item_text)
            
        self.console.append(f"<font color='green'>Manually added {len(files)} file(s) with dt={dt_val}!</font>")

    def add_unique_session_item(self, item_text):
        existing = [self.session_list_widget.item(i).text() for i in range(self.session_list_widget.count())]
        if item_text not in existing:
            self.session_list_widget.addItem(item_text)

    def setup_parameters_group(self, parent_layout):
        group_box = QGroupBox("Global Pipeline Parameters")
        layout = QVBoxLayout(group_box)

        # 1. Always Visible: dt
        dt_layout = QFormLayout()
        self.dt_input = QLineEdit("0.005")
        dt_layout.addRow("Global dt (bin size):", self.dt_input)
        layout.addLayout(dt_layout)

        # 2. The "Advanced" Toggle Button
        self.btn_toggle_advanced = QPushButton("Show Advanced Parameters")
        self.btn_toggle_advanced.setCheckable(True)
        self.btn_toggle_advanced.clicked.connect(self.toggle_advanced_parameters)
        layout.addWidget(self.btn_toggle_advanced)

        # 3. The Hidden Container
        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        
        self.out_dir_input = QLineEdit(r"C:\Users\PenPen\Desktop\Ferret\Code\Results")
        self.cache_path_input = QLineEdit(r"C:\Users\PenPen\Desktop\Ferret\Code\cache\preprocessed_data.pkl")
        self.unique_tones_input = QLineEdit(r"C:\Users\PenPen\Desktop\Ferret\Data\Bohan\unique_tones\unique_tones.npy")
        self.t_pre_input = QLineEdit("0.3")
        self.t_post_input = QLineEdit("0.3")

        common_group = QGroupBox("Shared Advanced Parameters")
        common_form = QFormLayout(common_group)
        common_form.addRow("Output Directory:", self.out_dir_input)
        common_form.addRow("Cache Path:", self.cache_path_input)
        common_form.addRow("Unique Tones Path:", self.unique_tones_input)
        common_form.addRow("T_PRE (s):", self.t_pre_input)
        common_form.addRow("T_POST (s):", self.t_post_input)

        advanced_layout.addWidget(common_group)

        # Start with it hidden
        self.advanced_container.setVisible(False)
        layout.addWidget(self.advanced_container)

        parent_layout.addWidget(group_box)

    def toggle_advanced_parameters(self):
        """Toggles the visibility of the advanced parameters container."""
        is_visible = self.advanced_container.isVisible()
        self.advanced_container.setVisible(not is_visible)
        
        if not is_visible:
            self.btn_toggle_advanced.setText("Hide Advanced Parameters")
        else:
            self.btn_toggle_advanced.setText("Show Advanced Parameters")

    def toggle_analyses_actions(self):
        """Toggles the visibility of the analyses actions container."""
        is_visible = self.analyses_container.isVisible()
        self.analyses_container.setVisible(not is_visible)

        if not is_visible:
            self.btn_toggle_analyses.setText("Hide Analyses")
        else:
            self.btn_toggle_analyses.setText("Show Analyses")

    def toggle_expanding_advanced(self):
        """Toggles visibility of Expanding N advanced parameter inputs."""
        is_visible = self.expanding_advanced_container.isVisible()
        self.expanding_advanced_container.setVisible(not is_visible)

        if not is_visible:
            self.btn_toggle_expanding_advanced.setText("Hide Advanced")
        else:
            self.btn_toggle_expanding_advanced.setText("Show Advanced")

    def toggle_chunks_advanced(self):
        """Toggles visibility of Chronological Chunks advanced parameter inputs."""
        is_visible = self.chunks_advanced_container.isVisible()
        self.chunks_advanced_container.setVisible(not is_visible)

        if not is_visible:
            self.btn_toggle_chunks_advanced.setText("Hide Advanced")
        else:
            self.btn_toggle_chunks_advanced.setText("Show Advanced")

    def setup_actions_group(self, parent_layout):
        group_box = QGroupBox("Execution Actions")
        layout = QVBoxLayout(group_box)

        self.btn_toggle_analyses = QPushButton("Show Analyses")
        self.btn_toggle_analyses.setCheckable(True)
        self.btn_toggle_analyses.clicked.connect(self.toggle_analyses_actions)
        layout.addWidget(self.btn_toggle_analyses)

        self.analyses_container = QWidget()
        analyses_layout = QVBoxLayout(self.analyses_container)

        self.btn_run_expanding = QPushButton("Run: First n Tracking vs Last n Playback")
        self.btn_run_expanding.clicked.connect(lambda: self.run_script("run_expanding_n.py"))

        self.n_values_input = QLineEdit("1, 2, 4, 8, 16, 32, 64, 128, 256")
        self.n_values_input.setMinimumWidth(260)

        expanding_row = QHBoxLayout()
        expanding_row.addWidget(self.btn_run_expanding)

        self.btn_toggle_expanding_advanced = QPushButton("Show Advanced")
        self.btn_toggle_expanding_advanced.setCheckable(True)
        self.btn_toggle_expanding_advanced.clicked.connect(self.toggle_expanding_advanced)

        self.expanding_advanced_container = QWidget()
        expanding_advanced_layout = QHBoxLayout(self.expanding_advanced_container)
        expanding_advanced_layout.setContentsMargins(20, 0, 0, 0)
        expanding_advanced_layout.addWidget(QLabel("N_VALUES (CSV):"))
        expanding_advanced_layout.addWidget(self.n_values_input)
        self.expanding_advanced_container.setVisible(False)
        
        self.btn_run_chunks = QPushButton("Run: Chunks Boxplot and Heatmap")
        self.btn_run_chunks.clicked.connect(lambda: self.run_script("run_chronological_chunks.py"))

        self.n_per_group_input = QLineEdit("5")
        self.n_per_group_input.setMaximumWidth(70)
        self.n_groups_input = QLineEdit("3")
        self.n_groups_input.setMaximumWidth(70)
        self.run_fdr_checkbox = QCheckBox("Run Pairwise FDR")
        self.run_fdr_checkbox.setChecked(True)

        chunks_row = QHBoxLayout()
        chunks_row.addWidget(self.btn_run_chunks)

        self.btn_toggle_chunks_advanced = QPushButton("Show Advanced")
        self.btn_toggle_chunks_advanced.setCheckable(True)
        self.btn_toggle_chunks_advanced.clicked.connect(self.toggle_chunks_advanced)

        self.chunks_advanced_container = QWidget()
        chunks_advanced_layout = QHBoxLayout(self.chunks_advanced_container)
        chunks_advanced_layout.setContentsMargins(20, 0, 0, 0)
        chunks_advanced_layout.addWidget(QLabel("N_PER_GROUP:"))
        chunks_advanced_layout.addWidget(self.n_per_group_input)
        chunks_advanced_layout.addWidget(QLabel("N_GROUPS:"))
        chunks_advanced_layout.addWidget(self.n_groups_input)
        chunks_advanced_layout.addWidget(self.run_fdr_checkbox)
        self.chunks_advanced_container.setVisible(False)

        self.btn_run_trigger_stats = QPushButton("Run: Trigger Stats")
        self.btn_run_trigger_stats.clicked.connect(lambda: self.run_script("run_trigger_stats.py"))

        self.btn_run_psth = QPushButton("Run: PSTH")
        self.btn_run_psth.clicked.connect(lambda: self.run_script("run_psth.py"))

        self.btn_run_first_last = QPushButton("Run: First vs Last Block PSTH")
        self.btn_run_first_last.clicked.connect(lambda: self.run_script("run_first_last_block.py"))

        self.btn_run_psth_matrix = QPushButton("Run: PSTH Matrix (Block × N)")
        self.btn_run_psth_matrix.clicked.connect(lambda: self.run_script("run_psth_matrix.py"))

        self.btn_run_transition = QPushButton("Run: Transition PSTH")
        self.btn_run_transition.clicked.connect(lambda: self.run_script("run_transition_psth.py"))

        self.btn_run_diagnostics = QPushButton("Run: Session Diagnostics")
        self.btn_run_diagnostics.clicked.connect(lambda: self.run_script("run_session_diagnostics.py"))

        # Keep the refresh button here as it relates to viewing the output
        self.btn_refresh = QPushButton("Refresh Results Folders")
        self.btn_refresh.clicked.connect(self.refresh_run_folders)

        analyses_layout.addLayout(expanding_row)
        analyses_layout.addWidget(self.btn_toggle_expanding_advanced)
        analyses_layout.addWidget(self.expanding_advanced_container)
        analyses_layout.addLayout(chunks_row)
        analyses_layout.addWidget(self.btn_toggle_chunks_advanced)
        analyses_layout.addWidget(self.chunks_advanced_container)
        analyses_layout.addWidget(self.btn_run_trigger_stats)
        analyses_layout.addWidget(self.btn_run_psth)
        analyses_layout.addWidget(self.btn_run_first_last)
        analyses_layout.addWidget(self.btn_run_psth_matrix)
        analyses_layout.addWidget(self.btn_run_transition)
        analyses_layout.addWidget(self.btn_run_diagnostics)
        analyses_layout.addWidget(self.btn_refresh)

        self.analyses_container.setVisible(False)
        layout.addWidget(self.analyses_container)

        parent_layout.addWidget(group_box)

    def delete_cache(self):
        cache_file = self.cache_path_input.text()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            self.console.append("<font color='orange'>Cache deleted. Next run will process raw data.</font>")
        else:
            self.console.append("No cache found to delete.")

    def write_config(self):
        """Reads all values from the GUI and writes them into config.py for the runners."""
        
        # 1. Format the session list for the file
        list_items = [self.session_list_widget.item(i).text() for i in range(self.session_list_widget.count())]
        config_dicts = []
        for item in list_items:
            if item.startswith("MANUAL"):
                parts = item.split(" | ")
                dt_val = float(parts[1].split("=")[1])
                data_file = parts[2].replace("\\", "\\\\") 
                config_dicts.append(f'{{"type": "manual", "data_file": r"{data_file}", "dt": {dt_val}}}')
            elif item.startswith("SCANNED"):
                folder = item.split(" | ")[1].replace("\\", "\\\\")
                config_dicts.append(f'{{"type": "scanned", "path": r"{folder}"}}')

        sessions_formatted = "[\n    " + ",\n    ".join(config_dicts) + "\n]"
        
        # 2. Extract numeric values from input boxes
        n_vals_list = [v.strip() for v in self.n_values_input.text().split(",")]
        n_vals_formatted = f"[{', '.join(n_vals_list)}]"

        # 3. Create the text content of the config file
        # Everything inside the f""" ... """ is what goes into the .py file
        config_content = f"""# AUTO-GENERATED BY GUI
import os
import numpy as np

# --- Paths ---
SESSION_CONFIGS = {sessions_formatted}

UNIQUE_TONES_PATH = r"{self.unique_tones_input.text()}"
OUTPUT_DIR = r"{self.out_dir_input.text()}"
CACHE_PATH = r"{self.cache_path_input.text()}"

# --- Signal Parameters ---
DT = {self.dt_input.text()}
T_PRE = {self.t_pre_input.text()}
T_POST = {self.t_post_input.text()}

N_BINS_PRE = int(T_PRE / DT)
N_BINS_POST = int(T_POST / DT)
EXPECTED_LENGTH = N_BINS_PRE + N_BINS_POST

BASELINE_BINS = int(0.2 / DT)
IDX_BASE_START = N_BINS_PRE - BASELINE_BINS
IDX_BASE_END = N_BINS_PRE
IDX_PEAK_START = N_BINS_PRE
IDX_PEAK_END = EXPECTED_LENGTH

# --- Analysis Parameters ---
N_VALUES = {n_vals_formatted}
N_PER_GROUP = {self.n_per_group_input.text()}
N_GROUPS = {self.n_groups_input.text()}

# --- NEW: Checkbox Value ---
RUN_CHUNK_FDR = {self.run_fdr_checkbox.isChecked()}
"""

        # 4. Save to disk
        with open("config.py", "w") as f:
            f.write(config_content)

    def run_script(self, script_name):
        if self.session_list_widget.count() == 0:
            QMessageBox.warning(self, "No Data", "Please add at least one session or file before running.")
            return
            
        if self.process.state() == QProcess.Running:
            QMessageBox.warning(self, "Warning", "A script is already running!")
            return

        self.write_config()
        self.console.clear()
        self.console.append(f"<b>--- Starting {script_name} ---</b>")
        
        self.btn_run_expanding.setEnabled(False)
        self.btn_run_chunks.setEnabled(False)
        self.btn_run_trigger_stats.setEnabled(False)
        self.btn_run_psth.setEnabled(False)

        # In a frozen build, run worker mode through this same executable.
        if getattr(sys, 'frozen', False):
            self.process.start(sys.executable, ["--run-script", script_name])
        else:
            self.process.start(sys.executable, [script_name])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        # Try UTF-8, fall back to system default (cp1252/cp850) if it fails
        try:
            stdout = bytes(data).decode("utf8")
        except UnicodeDecodeError:
            stdout = bytes(data).decode(sys.getfilesystemencoding(), errors='replace')
        
        self.console.append(stdout.strip())

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        try:
            stderr = bytes(data).decode("utf8")
        except UnicodeDecodeError:
            stderr = bytes(data).decode(sys.getfilesystemencoding(), errors='replace')
            
        # If we see a traceback, it's already formatting, otherwise wrap in red
        if "Traceback" in stderr:
            self.console.append(f"<font color='red'>{stderr.strip()}</font>")
        else:
            self.console.append(f"<font color='red'>{stderr.strip()}</font>")

    def process_finished(self):
        self.console.append("<b>--- Process Finished ---</b>")
        self.btn_run_expanding.setEnabled(True)
        self.btn_run_chunks.setEnabled(True)
        self.btn_run_trigger_stats.setEnabled(True)
        self.btn_run_psth.setEnabled(True)
        # Automatically scan for the newest folder and select it
        self.refresh_run_folders()

    def refresh_run_folders(self):
        """Scans the Output Directory for run folders, sorting by date/name properly."""
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        
        out_dir = self.out_dir_input.text()
        if not os.path.exists(out_dir):
            self.folder_combo.blockSignals(False)
            return

        # Get subdirectories
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        
        # Sort using the natural sort key (defined outside the class)
        subdirs.sort(key=natural_sort_key, reverse=True) 
        
        if subdirs:
            self.folder_combo.addItems(subdirs)
            
        self.folder_combo.addItem("[Root Output Folder]")
        self.folder_combo.blockSignals(False)
        self.refresh_file_list()

    def refresh_file_list(self):
        """Updates the list widget and clears the viewer to prevent wrong-file display."""
        self.file_list.clear()
        # Reset the viewer stack so you don't see the old file
        self.current_pixmap = None
        self.image_zoom = 1.0
        self.is_panning = False
        self.image_scroll.viewport().setCursor(Qt.ArrowCursor)
        self.image_label.clear()
        self.image_label.setText("Select a file to view")
        self.viewer_stack.setCurrentIndex(0)

        selected_folder = self.folder_combo.currentText()
        if not selected_folder:
            return
            
        base_path = os.path.abspath(self.out_dir_input.text())
        if selected_folder == "[Root Output Folder]":
            target_dir = base_path
        else:
            target_dir = os.path.join(base_path, selected_folder)
            
        if os.path.exists(target_dir):
            files = [f for f in os.listdir(target_dir) 
                     if os.path.isfile(os.path.join(target_dir, f)) 
                     and f.lower().endswith(('.png', '.csv', '.txt'))]
            
            # Apply our natural sort
            files.sort(key=natural_sort_key)
            self.file_list.addItems(files)

    def display_image(self, item):
        """Displays the selected file with strict path handling."""
        if not item or not item.text():
            return

        # 1. Clear current view to prevent "ghosting"
        self.current_pixmap = None
        self.image_zoom = 1.0
        self.is_panning = False
        self.image_scroll.viewport().setCursor(Qt.ArrowCursor)
        self.image_label.clear()
        self.csv_table.setRowCount(0)
        self.csv_table.setColumnCount(0)
        self.text_viewer.clear()

        # 2. Robust Path Construction
        filename = item.text()
        selected_folder = self.folder_combo.currentText()
        base_results_path = os.path.abspath(self.out_dir_input.text())

        if selected_folder == "[Root Output Folder]":
            full_path = os.path.join(base_results_path, filename)
        else:
            full_path = os.path.join(base_results_path, selected_folder, filename)

        full_path = os.path.normpath(full_path)
        
        # DEBUG: Check this in your console to verify the path is 100% correct
        print(f"DEBUG: Attempting to open -> {full_path}")

        if not os.path.exists(full_path):
            print(f"DEBUG: Path does not exist!")
            self.image_label.setText(f"File not found:\n{full_path}")
            self.viewer_stack.setCurrentIndex(0)
            return

        ext = filename.lower()

        # --- PNG HANDLING (Robust Version) ---
        if ext.endswith(('.png', '.jpg', '.jpeg')):
            self.viewer_stack.setCurrentIndex(0)
            pixmap = QPixmap(full_path)

            if pixmap.isNull():
                print(f"CRITICAL: Pixmap is NULL for {full_path}")
                self.image_label.setText("Error: Image could not be loaded.")
            else:
                self.current_pixmap = pixmap
                self.image_scroll.viewport().setCursor(Qt.OpenHandCursor)
                self.update_image_display()
                print(f"SUCCESS: Displaying image {filename}")

        # --- CSV HANDLING ---
        elif ext.endswith('.csv'):
            self.viewer_stack.setCurrentIndex(1)
            try:
                import csv
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = list(csv.reader(f))
                if data:
                    self.csv_table.setRowCount(len(data) - 1)
                    self.csv_table.setColumnCount(len(data[0]))
                    self.csv_table.setHorizontalHeaderLabels(data[0])
                    for r, row in enumerate(data[1:]):
                        for c, val in enumerate(row):
                            self.csv_table.setItem(r, c, QTableWidgetItem(str(val)))
                    self.csv_table.resizeColumnsToContents()
            except Exception as e:
                print(f"CSV Error: {e}")

        # --- TXT HANDLING ---
        elif ext.endswith('.txt'):
            self.viewer_stack.setCurrentIndex(2)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    self.text_viewer.setPlainText(f.read())
            except Exception as e:
                print(f"Text Error: {e}")

    def handle_current_file_changed(self, current, previous):
        """Ensures file preview updates whenever selection changes."""
        if current is not None:
            self.display_image(current)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

    def eventFilter(self, obj, event):
        if obj in (self.image_scroll.viewport(), self.image_label):
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
                and self.viewer_stack.currentIndex() == 0
                and self.current_pixmap is not None
            ):
                self.is_panning = True
                self.pan_last_global_pos = event.globalPosition()
                self.image_scroll.viewport().setCursor(Qt.ClosedHandCursor)
                return True

            if event.type() == QEvent.MouseMove and self.is_panning:
                delta = event.globalPosition() - self.pan_last_global_pos
                self.pan_last_global_pos = event.globalPosition()

                h_bar = self.image_scroll.horizontalScrollBar()
                v_bar = self.image_scroll.verticalScrollBar()
                h_bar.setValue(h_bar.value() - int(delta.x()))
                v_bar.setValue(v_bar.value() - int(delta.y()))
                return True

            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.is_panning:
                self.is_panning = False
                if self.current_pixmap is not None:
                    self.image_scroll.viewport().setCursor(Qt.OpenHandCursor)
                else:
                    self.image_scroll.viewport().setCursor(Qt.ArrowCursor)
                return True

        if event.type() == QEvent.Wheel and self.viewer_stack.currentIndex() == 0 and self.current_pixmap is not None:
            delta_y = event.angleDelta().y()
            if delta_y != 0:
                factor = self.image_zoom_step if delta_y > 0 else (1.0 / self.image_zoom_step)
                self.apply_image_zoom(factor, event.position())
                return True
        return super().eventFilter(obj, event)

    def apply_image_zoom(self, factor, viewport_pos=None):
        old_zoom = self.image_zoom
        new_zoom = max(self.image_zoom_min, min(self.image_zoom_max, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        h_bar = self.image_scroll.horizontalScrollBar()
        v_bar = self.image_scroll.verticalScrollBar()

        if viewport_pos is None:
            viewport_pos = self.image_scroll.viewport().rect().center()

        vp_x = float(viewport_pos.x())
        vp_y = float(viewport_pos.y())
        img_x_before = h_bar.value() + vp_x
        img_y_before = v_bar.value() + vp_y

        ratio = new_zoom / old_zoom
        self.image_zoom = new_zoom
        self.update_image_display()

        h_bar.setValue(int((img_x_before * ratio) - vp_x))
        v_bar.setValue(int((img_y_before * ratio) - vp_y))

    def update_image_display(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return

        target_size = self.image_scroll.viewport().size()
        if target_size.width() < 10 or target_size.height() < 10:
            target_size = QSize(800, 600)

        base_size = self.current_pixmap.size()
        base_size.scale(target_size, Qt.KeepAspectRatio)

        scaled_w = max(1, int(base_size.width() * self.image_zoom))
        scaled_h = max(1, int(base_size.height() * self.image_zoom))

        scaled_pixmap = self.current_pixmap.scaled(
            QSize(scaled_w, scaled_h),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

if __name__ == "__main__":
    # Worker mode for packaged executable
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        exit_code = execute_runner(sys.argv[2])
        sys.exit(exit_code)

    app = QApplication(sys.argv)
    window = FerretPipelineGUI()
    window.show()
    sys.exit(app.exec())