import time
from pathlib import Path

import matplotlib.pyplot as plt
import oursin as urchin
import json
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDesktopWidget,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QTextEdit,
)
from PyQt5.QtGui import QPixmap
import numpy as np
from cibrrig.plot import laser_colors
from PIL import Image
import seaborn as sns
from iblatlas.atlas import sph2cart, AllenAtlas
import asyncio

# Get absolute path of the file "brain_xyz.png"
brain_xyz_path = Path(__file__).parent / "brain_xyz.png"

ba = AllenAtlas()

# TODO: Fix ptich correctin for CCF


DEFAULT_SUBJECTS_PATH = Path("D:/Subjects")
DEFAULT_ARCHIVE_PATH = Path("U:/alf_data_repo/ramirez/Subjects")
DEFAULT_WORKING_PATH = Path("X:/alf_data_repo/ramirez/Subjects")

OCCIPITAL_APEX = (
    0,
    -6900,
    -600,
)  # Relative to bregma in microns in IBL coords (x,y,z)(ML,AP,DV)
OCCIPITAL_NADIR = (
    0,
    -6900,
    -3535,
)  # Relative to bregma in microns in IBL coords (x,y,z)(ML,AP,DV)
LAMBDA_IBL = (
    0,
    -4200,
    0,
)  # Relative to bregma in microns in IBL coords (x,y,z)(ML,AP,DV)
PITCH_CORRECTION = 0
COLORS = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00",
    "#ff00ff",
    "#00ffff",
    "#ff8000",
    "#ff0080",
    "#80ff00",
    "#80ff00",
]
INSERTION_TYPES = ["npx1.0", "npx2.0", "opto_200um", "opto_400um", "opto_600um"]

DEFAULT_WIRING = {
    "SYSTEM": "3B",
    "SYNC_WIRING_DIGITAL": {"P0.0": "record", "P0.7": "imec_sync"},
    "SYNC_WIRING_ANALOG": {
        "AI0": "diaphragm",
        "AI1": "ekg",
        "AI4": "pdiff",
        "AI5": "laser",
        "AI6": "diaphragm_integrated",
        "AI7": "temperature",
    },
}

# List of Sync inputs that are expected
POSSIBLE_WIRINGS = {
    "Digital": ["None", "record", "imec_sync", "right_camera", "Custom"],
    "Analog": [
        "None",
        "diaphragm",
        "ekg",
        "pdiff",
        "laser",
        "diaphragm_integrated",
        "temperature",
        "Custom",
    ],
}


def get_tip(x, y, z, d, phi, theta):
    """
    Ripped from iblatlas because we don't want to use the
    the atlas nor assume that the probe insertion point should be at the brain
    """
    return sph2cart(-d, theta, phi) + np.array((x, y, z))


def insertion2IBL(df):
    """
    Convert insertion coordinates to IBL coordinates
    dataframe must have the following columns:
    - Insertion ML (microns)
    - Insertion AP (microns)
    - Insertion DV (microns)
    - Reference ("occipital apex", "occipital nadir", "lambda", "bregma")

    Args:
        df (pd.DataFrame): DataFrame with insertion coordinates

    Returns:
        pd.DataFrame: DataFrame with IBL coordinates

    """
    print("converting to IBL")
    df["x"] = np.nan
    df["y"] = np.nan
    df["z"] = np.nan

    for i, row in df.iterrows():
        x = row["Insertion ML (microns)"]
        y = row["Insertion AP (microns)"]
        z = row["Insertion DV (microns)"]

        # Map x,y,z, to reference bregma
        if row["Reference"] == "occipital apex":
            x += OCCIPITAL_APEX[0]
            y += OCCIPITAL_APEX[1]
            z += OCCIPITAL_APEX[2]
        elif row["Reference"] == "occipital nadir":
            x += OCCIPITAL_NADIR[0]
            y += OCCIPITAL_NADIR[1]
            z += OCCIPITAL_NADIR[2]
        elif row["Reference"] == "lambda":
            x += LAMBDA_IBL[0]
            y += LAMBDA_IBL[1]
            z += LAMBDA_IBL[2]
        elif row["Reference"] == "bregma":
            pass
        else:
            raise ValueError(
                "Reference must be one of 'occipital apex', 'occipital nadir', 'lambda', 'bregma'"
            )

        df.loc[i, "x"] = x
        df.loc[i, "y"] = y
        df.loc[i, "z"] = z
    return df


def convert2ccf(df):
    """
    Convert IBL coordinates to CCF coordinates

    In IBL frame, bregma is at (0,0,0) and right is positive x, rostral is positive y, dorsal is positive z
    x,y,z -> ML, AP, DV

    In CCF frame, bregma is at (5400, 5739, 332) (AP,ML,DV) and right is positive x, caudal is positive y, ventral is positive z

    dataframe must have the following columns:
    - x
    - y
    - z
    - phi (azimuth/yaw)
    - theta (pitch/elevation)
    - Depth (microns)

    Args:
        df (pd.DataFrame): DataFrame with IBL coordinates

    Returns:
        pd.DataFrame: DataFrame with CCF coordinates
    """
    print("converting to ccf")
    df["AP tip CCF"] = np.nan
    df["ML tip CCF"] = np.nan
    df["DV tip CCF"] = np.nan
    df["phi CCF"] = np.nan
    df["theta CCF"] = np.nan

    for i, row in df.iterrows():
        x = row["x"]
        y = row["y"]
        z = row["z"]

        phi = row["phi (azimuth/yaw)"]
        theta = row["theta (pitch/elevation)"]
        depth = row["Depth (microns)"]

        theta += PITCH_CORRECTION
        tip = get_tip(x, y, z, depth, phi, theta)
        mlapdv = ba.xyz2ccf(tip / 1e6, mode="wrap")
        df.loc[i, "ML tip CCF"] = mlapdv[0]
        df.loc[i, "AP tip CCF"] = mlapdv[1]
        df.loc[i, "DV tip CCF"] = mlapdv[2]
        df.loc[i, "phi CCF"] = 270 - phi
        df.loc[i, "theta CCF"] = 90 - theta
    return df


def plot_probe_insertion(df, save_fn):
    try:
        asyncio.run(plot_probe_insertion_urchin(df, save_fn))
    except Exception:
        print("Error plotting with urchin, falling back to ibl/matplotlib")
        plot_probe_insertion_ibl(df, save_fn)


async def plot_probe_insertion_urchin(df, save_fn):
    ap = df["AP tip CCF"].values
    ml = df["ML tip CCF"].values
    dv = df["DV tip CCF"].values
    phi = df["phi CCF"].values
    theta = df["theta CCF"].values
    color = df["color"].values

    def opto_scale(x):
        return x / 2e3, 5, x / 2e3  # Convert from diameter in um to raidus in mm

    scales = []
    for i, row in df.iterrows():
        diameter = float(row["diameter"])
        if np.isnan(diameter):
            scale = [0.07, 3.84, 0.02]
        else:
            scale = opto_scale(diameter)
        scales.append(scale)

    n_probes = len(ap)
    positions = [(ap[i], ml[i], dv[i]) for i in range(n_probes)]
    angles = [(phi[i], theta[i], 0) for i in range(n_probes)]
    colors = [color[i] for i in range(n_probes)]

    delay = 1
    urchin.setup()
    time.sleep(1)
    urchin.ccf25.load()
    time.sleep(1)
    urchin.ccf25.grey.set_visibility(True)
    time.sleep(delay)
    urchin.ccf25.grey.set_material("transparent-unlit")
    time.sleep(delay)
    urchin.ccf25.grey.set_color("#000000")
    time.sleep(delay)
    urchin.ccf25.grey.set_alpha(0.1)
    time.sleep(delay)
    brain_areas = ["NTS", "VII", "AMB", "LRNm"]
    area_list = urchin.ccf25.get_areas(brain_areas)
    urchin.ccf25.set_visibilities(area_list, True)
    urchin.ccf25.set_materials(area_list, "transparent-unlit")
    urchin.ccf25.set_colors(area_list, ["#000000"] * len(area_list))
    urchin.ccf25.set_alphas(area_list, 0.4)

    time.sleep(delay)
    urchin.camera.main.set_zoom(7)
    time.sleep(delay)
    probes = urchin.probes.create(n_probes)
    # time.sleep(delay)
    urchin.probes.set_positions(probes, positions)
    # time.sleep(delay)
    urchin.probes.set_angles(probes, angles)
    # time.sleep(delay)
    urchin.probes.set_colors(probes, colors)
    # time.sleep(delay)
    urchin.probes.set_scales(probes, scales)
    # time.sleep(delay)

    urchin.camera.main.set_rotation((-80, 140, 0))
    time.sleep(delay)

    a_cam = urchin.camera.Camera()
    s_cam = urchin.camera.Camera()
    c_cam = urchin.camera.Camera()

    a_cam.set_rotation("axial")
    a_cam.set_zoom(7)

    s_cam.set_rotation("sagittal")
    s_cam.set_zoom(5)

    c_cam.set_rotation("coronal")
    c_cam.set_zoom(5)

    wd = 1800
    ht = 1200

    angled_png = await urchin.camera.main.screenshot(size=[wd, ht])
    time.sleep(0.1)
    axial_png = await a_cam.screenshot(size=[wd, ht])
    time.sleep(0.1)
    sagittal_png = await s_cam.screenshot(size=[wd, ht])
    time.sleep(0.1)
    coronal_png = await c_cam.screenshot(size=[wd, ht])
    time.sleep(0.1)

    grid_image = Image.new("RGBA", (wd * 2, ht * 2))

    grid_image.paste(angled_png, (0, 0))
    grid_image.paste(sagittal_png, (wd, 0))
    grid_image.paste(axial_png, (0, ht))
    grid_image.paste(coronal_png, (wd, ht))

    grid_image.save(save_fn)


def plot_probe_insertion_ibl(df, save_fn):
    xyz = df[["x", "y", "z"]].values
    tip = get_tip(
        df["x"],
        df["y"],
        df["z"],
        df["Depth (microns)"],
        df["phi (azimuth/yaw)"],
        df["theta (pitch/elevation)"],
    )
    tip = tip.T
    colors = df["color"].values
    n_lines = xyz.shape[0]

    # Make a line from xyz to tip for each row
    f, ax = plt.subplots(2, 2)
    ba.plot_cslice(0, volume="boundary", ax=ax[0, 0], mapping="Cosmos", alpha=0.3)
    ba.plot_sslice(0, volume="boundary", ax=ax[1, 0], mapping="Cosmos", alpha=0.3)
    ba.plot_hslice(
        -5000 / 1e6, volume="boundary", ax=ax[0, 1], mapping="Cosmos", alpha=0.3
    )
    ax[1, 1].axis("off")
    for ii in range(n_lines):
        x = [xyz[ii, 0], tip[ii, 0]]
        y = [xyz[ii, 1], tip[ii, 1]]
        z = [xyz[ii, 2], tip[ii, 2]]
        ax[0, 0].plot(x, z, color=colors[ii], marker=".")
        ax[1, 0].plot(y, z, color=colors[ii], marker=".")
        ax[0, 1].plot(x, y, color=colors[ii], marker=".")

    ax[0, 0].set_xlabel(r"ML $\mu$m")
    ax[0, 0].set_ylabel(r"DV $\mu$m")

    ax[1, 0].set_xlabel(r"AP $\mu$m")
    ax[1, 0].set_ylabel(r"DV $\mu$m")

    ax[0, 1].set_xlabel(r"ML $\mu$m")
    ax[0, 1].set_ylabel(r"AP $\mu$m")

    ax[0, 0].set_title("Coronal projection")
    ax[1, 0].set_title("Sagittal projection")
    ax[0, 1].set_title("Axial projection")

    ax[0, 0].text(
        1, 1, "AP=0", color="red", transform=ax[0, 0].transAxes, ha="right", va="top"
    )
    ax[1, 0].text(
        1, 1, "ML=0", color="red", transform=ax[1, 0].transAxes, ha="right", va="top"
    )
    ax[0, 1].text(
        1,
        1,
        "DV=-500um",
        color="red",
        transform=ax[0, 1].transAxes,
        ha="right",
        va="top",
    )

    plt.savefig(save_fn, dpi=300)
    plt.close("all")


def plot_insertion_layout(df, save_fn):
    ref_list = ["occipital apex", "occipital nadir"]
    df = df.query("Reference in @ref_list")
    f = plt.figure()
    ax = f.add_subplot(111)
    for i, row in df.iterrows():
        insertion_num = int(row["Insertion number"])
        gate = row["Gate"]
        ml = row["Insertion ML (microns)"]
        dv = row["Insertion DV (microns)"]
        color = row["color"]
        probe = row["probe"]
        s = f"{probe} - insertion {insertion_num} - {gate}"
        ax.text(ml, dv, s, color=color, ha="left", va="bottom")
        ax.plot(ml, dv, "o", color=color)
    plt.xlabel("Insertion ML (microns)")
    plt.ylabel("Insertion DV (microns)")

    ax.invert_yaxis()
    ax.axis("equal")

    plt.title("Caudal Approach Layout Coronal Projection")
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(save_fn, dpi=300)
    plt.close("all")


class DirectorySelector(QWidget):
    """
    GUI to select paths for backup. Expands QWidget.

    This class provides a graphical user interface (GUI) for selecting local and remote paths for backup,
    as well as options for removing optogenetic artifacts and running ephys quality control.

    Attributes:
        local_run_path (Path): Path to the local run directory.
        remote_archive_path (Path): Path to the remote archive directory.
        remote_working_path (Path): Path to the remote working directory.
        remove_opto_artifact (bool): Option to remove optogenetic artifacts.
        run_ephys_qc (bool): Option to run ephys quality

    Args:
        QWidget (QWidget): Base class QWidget.
    """

    def __init__(self):
        """
        Set default settings and initialize the UI.
        """
        super().__init__()
        self.local_run_path = DEFAULT_SUBJECTS_PATH
        self.remote_archive_path = DEFAULT_ARCHIVE_PATH
        self.remote_working_path = DEFAULT_WORKING_PATH
        self.remove_opto_artifact = False
        self.run_ephys_qc = True

        self.init_ui()

    def init_ui(self):
        """
        Initialize the UI layout.
        """
        layout = QVBoxLayout()

        grid_layout = QGridLayout()

        # Local run path selection
        self.local_run_label = QLabel("Local Run Path:")
        self.local_run_line_edit = QLineEdit(str(self.local_run_path))
        self.local_run_button = QPushButton("Browse...")
        self.local_run_button.clicked.connect(self.select_local_run_path)
        grid_layout.addWidget(self.local_run_label, 0, 0)
        grid_layout.addWidget(self.local_run_line_edit, 0, 1)
        grid_layout.addWidget(self.local_run_button, 0, 2)

        # Remote archive path selection
        self.remote_archive_label = QLabel("Remote Subjects Archive Path:")
        self.remote_archive_line_edit = QLineEdit(str(self.remote_archive_path))
        self.remote_archive_button = QPushButton("Browse...")
        self.remote_archive_button.clicked.connect(self.select_remote_archive_path)
        grid_layout.addWidget(self.remote_archive_label, 1, 0)
        grid_layout.addWidget(self.remote_archive_line_edit, 1, 1)
        grid_layout.addWidget(self.remote_archive_button, 1, 2)

        # Remote working path selection
        self.remote_working_label = QLabel("Remote Working Subjects Path:")
        self.remote_working_line_edit = QLineEdit(str(self.remote_working_path))
        self.remote_working_button = QPushButton("Browse...")
        self.remote_working_button.clicked.connect(self.select_remote_working_path)
        grid_layout.addWidget(self.remote_working_label, 2, 0)
        grid_layout.addWidget(self.remote_working_line_edit, 2, 1)
        grid_layout.addWidget(self.remote_working_button, 2, 2)

        # Checkbox for remove_opto_artifact
        self.remove_opto_artifact_checkbox = QCheckBox("Remove Opto Artifact")
        self.remove_opto_artifact_checkbox.setChecked(self.remove_opto_artifact)
        self.remove_opto_artifact_checkbox.setToolTip(
            "Uncheck if stimulation is unlikely to cause light artifacts"
        )
        self.remove_opto_artifact_checkbox.stateChanged.connect(
            self.toggle_remove_opto_artifact
        )
        grid_layout.addWidget(self.remove_opto_artifact_checkbox, 3, 0, 1, 3)

        # Checkbox for run_ephys_qc
        self.run_ephys_qc_checkbox = QCheckBox("Run EphysQC")
        self.run_ephys_qc_checkbox.setChecked(self.run_ephys_qc)
        grid_layout.addWidget(self.run_ephys_qc_checkbox, 4, 0, 1, 3)

        # Spin box for number of probes
        self.num_probes_label = QLabel("Number of Probes:")
        self.num_probes_spinbox = QSpinBox()
        self.num_probes_spinbox.setValue(1)
        self.num_probes_spinbox.setMinimum(0)
        grid_layout.addWidget(self.num_probes_label, 5, 0)
        grid_layout.addWidget(self.num_probes_spinbox, 5, 1)

        # Spin box for number of opto fibers
        self.num_opto_fibers_label = QLabel("Number of Opto Fibers:")
        self.num_opto_fibers_spinbox = QSpinBox()
        self.num_opto_fibers_spinbox.setValue(0)
        self.num_opto_fibers_spinbox.setMinimum(0)
        grid_layout.addWidget(self.num_opto_fibers_label, 6, 0)
        grid_layout.addWidget(self.num_opto_fibers_spinbox, 6, 1)

        layout.addLayout(grid_layout)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
        self.setWindowTitle("Directory Selector")

    def select_local_run_path(self):
        """Select the local run path with UI"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Local Run Path", str(self.local_run_path)
        )
        if directory:
            self.local_run_path = Path(directory)
            self.local_run_line_edit.setText(str(self.local_run_path))
            self.get_session_info()

    def select_remote_archive_path(self):
        """Select the remote archive path with UI"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Remote Subjects Archive Path", str(self.remote_archive_path)
        )
        if directory:
            self.remote_archive_path = Path(directory)
            self.remote_archive_line_edit.setText(str(self.remote_archive_path))

    def select_remote_working_path(self):
        """Select the remote working path with UI"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Remote Subjects Working Path", str(self.remote_working_path)
        )
        if directory:
            self.remote_working_path = Path(directory)
            self.remote_working_line_edit.setText(str(self.remote_working_path))

    def toggle_remove_opto_artifact(self, state):
        """Toggle the remove_opto_artifact checkbox"""
        if state == Qt.Checked:
            self.remove_opto_artifact = True
        else:
            self.remove_opto_artifact = False

    def get_gates(self):
        gate_paths = list(self.local_run_path.glob("*_g[0-9]*"))
        # order by gate number
        gate_nums = [int(str(gate).split("_g")[-1]) for gate in gate_paths]
        self.gate_paths = [x for _, x in sorted(zip(gate_nums, gate_paths))]
        self.n_gates = len(self.gate_paths)

    def infer_num_probes(self):
        self.num_probes = 0
        for gate in self.gate_paths:
            probes = list(gate.glob("*imec*"))
            self.num_probes = max(len(probes), self.num_probes)
        # Update the spinbox value
        self.num_probes_spinbox.setValue(self.num_probes)

    def get_session_info(self):
        self.get_gates()
        self.infer_num_probes()

    def submit(self):
        if self.local_run_path == DEFAULT_SUBJECTS_PATH:
            print(
                "You picked the root Subjects folder. This is a scary thing to do and incorrect. Please select a run folder."
            )
            return None
        else:
            self.close()  # Close the GUI window

    def get_paths(self):
        """Return the selected paths and options"""
        return (
            self.local_run_path,
            self.remote_archive_path,
            self.remote_working_path,
            self.remove_opto_artifact,
            self.run_ephys_qc,
            self.gate_paths,
            self.num_probes_spinbox.value(),
            self.num_opto_fibers_spinbox.value(),
        )


class OptoFileFinder(QDialog):
    """
    Dialog box to select the opto_calibration.json file if it is not found

    Attributes:
        opto_file (Path): Path to the opto_calibration.json file
    """

    opto_file_selected = pyqtSignal(Path)

    def __init__(self, title=""):
        """Initialize the dialog box"""
        super().__init__()
        self.setWindowTitle("File Selection")
        self.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout()

        label = QLabel(
            f"opto_calibration.json not found for {title}. Please select a file or skip."
        )
        layout.addWidget(label)

        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        skip_button = QPushButton("Skip")
        skip_button.clicked.connect(self.skip_file)
        layout.addWidget(skip_button)

        self.setLayout(layout)

        self.opto_file = None  # Initialize opto_file attribute

    def select_file(self):
        """Select the opto_calibration.json file with UI"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.opto_file = Path(file_path)
            self.opto_file_selected.emit(
                self.opto_file
            )  # Emit signal with opto file path
            self.close()

    def skip_file(self):
        """Skip the opto_calibration.json file"""
        self.opto_file = Path("")
        self.opto_file_selected.emit(self.opto_file)  # Emit signal with None (skipped)
        self.close()

    def get_opto_file(self):  # Method to get opto file after dialog is closed
        return self.opto_file


class WiringEditor(QDialog):
    """
    Dialog box to select the wiring file

    Attributes:
        output_wiring (dict): Dictionary of the selected wiring
        digital_entries (dict): Dictionary of the digital mapping (channels: signals)
        analog_entries (dict): Dictionary of the analog mapping (channels: signals)

    """

    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.initUI()

    def initUI(self):
        """Initialize the UI layout"""
        self.setWindowTitle(f"Wiring Editor for {self.title}")
        self.setGeometry(100, 100, 600, 200)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Digital and Analog Groups
        groups_layout = QHBoxLayout()
        main_layout.addLayout(groups_layout)

        # Digital Group
        digital_group = QGroupBox("Digital")
        digital_layout = QVBoxLayout()

        # Create a combo box for each digital entry
        self.digital_entries = {}
        digital_keys = [f"P0.{i}" for i in range(8)]
        for key in digital_keys:
            label = QLabel(key)
            combo_box = QComboBox()
            combo_box.addItems(POSSIBLE_WIRINGS.get("Digital", []))
            combo_box.setCurrentText(
                DEFAULT_WIRING.get("SYNC_WIRING_DIGITAL", {}).get(key, "None")
            )
            combo_box.currentIndexChanged.connect(
                lambda _, cb=combo_box, k=key: self.on_digital_value_changed(cb, k)
            )
            digital_layout.addWidget(label)
            digital_layout.addWidget(combo_box)
            self.digital_entries[key] = combo_box

        digital_group.setLayout(digital_layout)
        groups_layout.addWidget(digital_group)

        # Analog Group
        analog_group = QGroupBox("Analog")
        analog_layout = QVBoxLayout()

        # Create a combo box for each analog entry
        self.analog_entries = {}
        analog_keys = [f"AI{i}" for i in range(8)]
        for key in analog_keys:
            label = QLabel(key)
            combo_box = QComboBox()
            combo_box.addItems(POSSIBLE_WIRINGS.get("Analog", []))
            combo_box.setCurrentText(
                DEFAULT_WIRING.get("SYNC_WIRING_ANALOG", {}).get(key, "None")
            )
            combo_box.currentIndexChanged.connect(
                lambda _, cb=combo_box, k=key: self.on_analog_value_changed(cb, k)
            )
            analog_layout.addWidget(label)
            analog_layout.addWidget(combo_box)
            self.analog_entries[key] = combo_box

        analog_group.setLayout(analog_layout)
        groups_layout.addWidget(analog_group)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_values)
        main_layout.addWidget(save_button)

    def on_digital_value_changed(self, combo_box, key):
        """Update the digital value"""
        if combo_box.currentText() == "Custom":
            text, ok = QInputDialog.getText(
                self, "Custom Value", f"Enter custom value for {key}:"
            )
            if ok:
                combo_box.addItem(text)
                combo_box.setCurrentText(text)

    def on_analog_value_changed(self, combo_box, key):
        """Update the analog value"""
        if combo_box.currentText() == "Custom":
            text, ok = QInputDialog.getText(
                self, "Custom Value", f"Enter custom value for {key}:"
            )
            if ok:
                combo_box.addItem(text)
                combo_box.setCurrentText(text)

    def save_values(self):
        """Save the output wiring attributes"""
        output_dictionary = {
            "SYSTEM": "3B",
            "SYNC_WIRING_ANALOG": {},
            "SYNC_WIRING_DIGITAL": {},
        }

        # Digital mappings
        for key, combo_box in self.digital_entries.items():
            value = combo_box.currentText()
            if value != "None":
                output_dictionary["SYNC_WIRING_DIGITAL"][key] = value

        # Analog mappings
        for key, combo_box in self.analog_entries.items():
            value = combo_box.currentText()
            if value != "None":
                output_dictionary["SYNC_WIRING_ANALOG"][key] = value

        # Print and save the output dictionary
        print("Output Dictionary:", output_dictionary)
        self.output_wiring = output_dictionary
        self.close()

    def get_output_wiring(self):
        return self.output_wiring


class InsertionTableAppBase(QDialog):
    def __init__(self, n_rows=1, n_gates=20, name="", save_fn=None):
        super().__init__()
        self.setWindowTitle(f"Insertion Data Table {name}")
        self.n_gates = n_gates
        self.save_fn = save_fn
        self.name = name
        self.df = None

        # Load data if it exists
        if (self.save_fn is not None) and (self.save_fn.exists()):
            self.load_csv()
            self.n_rows = len(self.df)
        else:
            self.n_rows = n_rows

        # Get screen size and set window size relative to it
        screen = QDesktopWidget().screenGeometry()
        width, height = int(screen.width() * 0.9), int(screen.height() * 0.5)
        self.setGeometry(100, 100, width, height)

        # Create central widget and layout
        main_layout = QHBoxLayout(self)

        # Create a vertical layout for the table and buttons
        table_layout = QVBoxLayout()

        # Add information label
        info = f"""
            LOG ALL INSERTIONS FOR {name}, 
            INCLUDING THOSE WITHOUT ASSOCIATED RECORDINGS.\n
            Record insertion position relative to the chosen reference point\n
            IF MULTIPLE GATES EXIST FOR THE SAME INSERTION,REPEAT THE INSERTION NUMBER BUT SET THE GATE APPROPRIATELY.\n
            ML: LEFT is negative\n
            AP: ROSTRAL is positive\n
            DV: DORSAL is positive\n
            DEPTH SHOULD BE POSITIVE (i.e., distance into the brain)"
        """
        info_label = QLabel(info)
        info_label.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(info_label)

        self.get_headers()
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.headers))
        self.table.setRowCount(self.n_rows)

        # Set headers
        self.table.setHorizontalHeaderLabels(self.headers)

        # Fill insertion numbers and setup columns
        for row in range(self.n_rows):
            self.add_row(row)

        # Adjust column widths to fit the window
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        table_layout.addWidget(self.table)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # Add export button
        export_button = QPushButton("Export to DataFrame")
        export_button.setStyleSheet(
            "background-color: #f5e6ab; color: black; font-weight: bold;"
        )
        export_button.clicked.connect(self.create_dataframe)
        button_layout.addWidget(export_button)

        # Add button to add another row
        add_row_button = QPushButton("Add Row")
        add_row_button.clicked.connect(self.add_new_row)
        button_layout.addWidget(add_row_button)

        # Add button to delete the last row
        delete_row_button = QPushButton("Delete Last Row")
        delete_row_button.clicked.connect(self.delete_last_row)
        button_layout.addWidget(delete_row_button)

        # Add button to add another column
        add_column_button = QPushButton("Add Column")
        add_column_button.clicked.connect(self.add_new_column)
        button_layout.addWidget(add_column_button)

        # Add the button layout to the table layout
        table_layout.addLayout(button_layout)

        # Add the table layout to the main layout
        main_layout.addLayout(table_layout)

        # Add image to the right of the table
        image_label = QLabel()
        pixmap = QPixmap(str(brain_xyz_path))
        image_label.setPixmap(pixmap)
        main_layout.addWidget(image_label)

        if self.df is not None:
            self.populate_table()


    def get_headers(self):
        headers = [
            "Insertion number",
            "Gate",
            "Reference",
            "Insertion ML (microns)",
            "Insertion AP (microns)",
            "Insertion DV (microns)",
            "Depth (microns)",
            "phi (azimuth/yaw)",
            "theta (pitch/elevation)",
            "Insertion Type",
        ]
        self.gate_column = 1
        self.reference_column = 2
        self.numeric_columns = [3, 4, 5, 6, 7, 8]
        self.phi_column = 7
        self.theta_column = 8
        self.insertion_type_column = 9
        self.numeric_headers = headers[3:9]
        self.headers = headers

        return headers

    def add_row(self, row):
        # Insertion number (modifiable)
        insertion_item = QTableWidgetItem(str(row))
        insertion_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, insertion_item)

        # AP, DV, ML columns (integer input)
        for col in self.numeric_columns:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        # Reference column (dropdown)
        reference_combo = QComboBox()
        reference_options = ["bregma", "lambda", "occipital apex", "occipital nadir"]
        reference_combo.addItems(reference_options)
        reference_combo.currentTextChanged.connect(lambda: self.update_angles(row))
        self.table.setCellWidget(row, self.reference_column, reference_combo)

        # Gate column (dropdown)
        gate_combo = QComboBox()
        gate_options = ["dnr"] + [f"g{i}" for i in range(0, self.n_gates)]
        gate_combo.addItems(gate_options)
        self.table.setCellWidget(row, self.gate_column, gate_combo)

        # Insertion type column (dropdown)
        insertion_type_combo = QComboBox()
        insertion_type_combo.addItems(self.get_insertion_types())
        self.table.setCellWidget(row, self.insertion_type_column, insertion_type_combo)

        # Set default angles for "occipital apex"
        self.update_angles(row)

    def get_insertion_types(self):
        return INSERTION_TYPES

    def update_angles(self, row):
        reference_combo = self.table.cellWidget(row, self.reference_column)
        reference = reference_combo.currentText()
        if reference in ["occipital apex", "occipital nadir"]:
            phi_item = QTableWidgetItem("270")
            phi_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.phi_column, phi_item)
            theta_item = QTableWidgetItem("90")
            theta_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.theta_column, theta_item)
        elif reference in ["bregma", "lambda"]:
            phi_item = QTableWidgetItem("0")
            phi_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.phi_column, phi_item)
            theta_item = QTableWidgetItem("0")
            theta_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.theta_column, theta_item)

    def add_new_row(self):
        current_row_count = self.table.rowCount()
        self.table.insertRow(current_row_count)
        self.add_row(current_row_count)

        # Copy the previous row's data
        if current_row_count > 0:
            self.copy_previous_row(current_row_count)

        # Increment insertion number
        self.table.item(current_row_count, 0).setText(str(current_row_count))

    def delete_last_row(self):
        current_row_count = self.table.rowCount()
        if current_row_count > 0:
            self.table.removeRow(current_row_count - 1)

    def copy_previous_row(self, current_row_count):
        for col in range(1, self.table.columnCount()):
            previous_item = self.table.item(current_row_count - 1, col)
            if previous_item:
                new_item = QTableWidgetItem(previous_item.text())
                new_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(current_row_count, col, new_item)
            else:
                previous_widget = self.table.cellWidget(current_row_count - 1, col)
                if previous_widget:
                    if isinstance(previous_widget, QComboBox):
                        new_widget = QComboBox()
                        for i in range(previous_widget.count()):
                            new_widget.addItem(previous_widget.itemText(i))
                        new_widget.setCurrentIndex(previous_widget.currentIndex())
                        self.table.setCellWidget(current_row_count, col, new_widget)

    def add_new_column(self):
        column_name, ok = QInputDialog.getText(
            self, "Column Name", "Enter the name of the new column:"
        )
        if ok and column_name:
            current_column_count = self.table.columnCount()
            self.table.insertColumn(current_column_count)
            header_item = QTableWidgetItem(column_name)
            self.table.setHorizontalHeaderItem(current_column_count, header_item)
            for row in range(self.table.rowCount()):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, current_column_count, item)

    def export_to_dataframe(self):
        data = {
            self.table.horizontalHeaderItem(col).text(): []
            for col in range(self.table.columnCount())
        }
        for row in range(self.table.rowCount()):
            row_data = []
            for col in range(self.table.columnCount()):
                if (
                    col == self.gate_column
                    or col == self.reference_column
                    or col == self.insertion_type_column
                ):  # Gate, Reference, and Insertion Type columns (dropdown)
                    widget = self.table.cellWidget(row, col)
                    if widget is not None:
                        row_data.append(widget.currentText())
                    else:
                        row_data.append("")
                else:
                    item = self.table.item(row, col)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
            for header, value in zip(data.keys(), row_data):
                data[header].append(value)
        df = pd.DataFrame(data)
        df[self.numeric_headers] = df[self.numeric_headers].apply(
            pd.to_numeric, errors="coerce"
        )
        # Drop rows where all numeric columns are NaN
        df = df.dropna(subset=self.numeric_headers, how="any")
        df["probe"] = self.name
        self.df = df
        print(df)

    def convert_and_export(self):
        self.df = insertion2IBL(self.df)
        self.df = convert2ccf(self.df)

    def create_dataframe(self):
        self.export_to_dataframe()
        self.convert_and_export()
        self.close()

    def get_insertions(self):
        return self.df

    def to_csv(self):
        self.df.to_csv(self.save_fn, index=False)

    # Make a method to load from a csv if it exists
    def load_csv(self):
        """
        Load a csv and populate the table if it exists
        """
        df = pd.read_csv(self.save_fn)
        self.df = df

    def populate_table(self):
        """
        Populate the table with data from a dataframe
        """
        df = self.df
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if j in [self.gate_column, self.reference_column, self.insertion_type_column]:
                    # Set the current text for the QComboBox widgets
                    widget = self.table.cellWidget(i, j)
                    if widget is not None and isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))
                else:
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, j, item)


class NpxInsertionTableApp(InsertionTableAppBase):
    def __init__(self, n_rows=1, n_gates=10, name="", save_fn=None):
        super().__init__(n_rows, n_gates, name, save_fn)

    def get_insertion_types(self):
        return ["npx1.0", "npx2.0"]

    def add_row(self, row):
        super().add_row(row)
        reference_combo = self.table.cellWidget(row, self.reference_column)
        reference_combo.setCurrentText("occipital apex")
        insertion_type_combo = self.table.cellWidget(row, self.insertion_type_column)
        insertion_type_combo.setCurrentText("npx1.0")

    def convert_and_export(self):
        # Call the base class method
        super().convert_and_export()

        # Replace the color with the insertion number
        self.df["color"] = self.df["Insertion number"].apply(
            lambda x: COLORS[int(x) % len(COLORS)]
        )
        self.df["diameter"] = np.nan


class OptoInsertionTableApp(InsertionTableAppBase):
    def __init__(self, n_rows=1, n_gates=10, name="", save_fn=None):
        super().__init__(n_rows, n_gates, name, save_fn)
        self.setStyleSheet("background-color: lightblue;")
        header = self.table.horizontalHeader()
        header.setStyleSheet(
            "QHeaderView::section { background-color: lightblue; color: black; }"
        )

    def get_headers(self):
        headers = super().get_headers()
        headers.extend(["wavelength", "diameter"])
        return headers

    def get_insertion_types(self):
        return ["opto"]

    def add_row(self, row):
        super().add_row(row)
        reference_combo = self.table.cellWidget(row, self.reference_column)
        reference_combo.setCurrentText("bregma")
        insertion_type_combo = self.table.cellWidget(row, self.insertion_type_column)
        insertion_type_combo.setCurrentText("opto_200um")

        # Add wavelength and diameter columns with default values
        wavelength_item = QTableWidgetItem("473")
        wavelength_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.table.columnCount() - 2, wavelength_item)

        diameter_item = QTableWidgetItem("200")
        diameter_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.table.columnCount() - 1, diameter_item)

    # Modify the inherited convert2ccf method to replace the color with the wavelength
    def convert_and_export(self):
        # Call the base class method
        super().convert_and_export()

        # Replace the color with the wavelength
        self.df["wavelength"] = self.df["wavelength"].apply(
            pd.to_numeric, errors="coerce"
        )
        self.df["color"] = self.df["wavelength"].apply(
            lambda x: laser_colors.get(x, "#000000")
        )


class NotesDialog(QDialog):
    def __init__(self, n_gates, notes_fn):
        super().__init__()
        self.setWindowTitle("Notes Dialog")
        self.notes_fn = notes_fn

        # initialize the window to be big
        self.setGeometry(100, 100, 1000, 800)

        # Create layout
        layout = QVBoxLayout(self)

        # Create a vertical layout for the overall notes
        overall_notes_layout = QHBoxLayout()

        # Add overall notes label and text field
        overall_notes_label = QLabel("Overall Notes:\nInjections\nProtocol\netc.")
        overall_notes_layout.addWidget(overall_notes_label)
        self.overall_notes_text = QTextEdit()  # Use QTextEdit for larger text area
        overall_notes_layout.addWidget(self.overall_notes_text)

        # Add the overall notes layout to the main layout
        layout.addLayout(overall_notes_layout)

        # Create a vertical layout for the gate notes
        gate_notes_layout = QHBoxLayout()

        # Add text fields to the gate notes layout
        self.text_fields = []
        for i in range(n_gates):
            label = QLabel(f"Gate {i} Notes:")
            text_field = QTextEdit()  # Use QTextEdit for larger text area
            self.text_fields.append(text_field)
            gate_notes_layout.addWidget(label)
            gate_notes_layout.addWidget(text_field)

        # Add the gate notes layout to the main layout
        layout.addLayout(gate_notes_layout)

        # Add submit button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit)
        layout.addWidget(submit_button)

        # Load notes if the file exists
        self.load_notes()

    def submit(self):
        self.close()  # Close the dialog box

    def save_notes(self):
        # Save notes in JSON form
        notes = {
            "overall_notes": self.overall_notes_text.toPlainText(),
            "gate_notes": {
                f"gate_{i}": text_field.toPlainText()
                for i, text_field in enumerate(self.text_fields)
            },
        }
        with open(self.notes_fn, "w") as f:
            json.dump(notes, f)

        return notes

    def load_notes(self):
        if self.notes_fn.exists():
            with open(self.notes_fn, "r") as f:
                notes = json.load(f)
                self.overall_notes_text.setText(notes.get("overall_notes", ""))
                for i, text_field in enumerate(self.text_fields):
                    gate_note = notes.get("gate_notes", {}).get(f"gate_{i}", "")
                    text_field.setText(gate_note)
