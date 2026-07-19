"""Desktop GUI for Dehazer, built with PySide6.

Provides a drag-and-drop interface to run the dehazing pipeline, a processing
queue, and two comparison views (image-to-image and pipeline-to-pipeline).
Launched via ``python -m dehazer``.
"""

import json
import logging
import os
import threading

import numpy as np
from PySide6.QtCore import Qt, QMimeData, QSize
from PySide6.QtGui import QPixmap, QIcon, QImage, QTextCursor, QDrag, QColor
from PySide6.QtWidgets import *

from .config import OUTPUT_DIR, ensure_output_dir
from .core import dehaze, dehazer_data
from .image_diff import ImageComparator
from .u_guided_filter import guided_filter, guided_filter_data                  # kwargs | r : int, eps : float
from .u_soft_matting import soft_matting, soft_matting_data                     # kwargs | maxiter : int, win_radius : int, eps : int, lam : int, max_processes : int
from .u_soft_matting_chunked import chunked_soft_matting, chunk_soft_matting_data    # kwargs | maxiter : int, n_cut_width : int, n_cut_height : int, win_radius : int, eps : float, lam : float, max_processes : int, ratio : float


class QueueWidget(QWidget):
    """Two-pane list showing pending and completed dehazing tasks."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # --- Left list: pending ---
        self.pending_list = QListWidget()
        self.pending_list.setIconSize(QSize(80, 80))
        self.pending_list.setAlternatingRowColors(True)
        self.pending_list.setSpacing(6)
        layout.addWidget(self.pending_list)

        # --- Right list: completed ---
        self.completed_list = QListWidget()
        self.completed_list.itemActivated.connect(self.view_images)
        self.completed_list.setIconSize(QSize(80, 80))
        self.completed_list.setAlternatingRowColors(True)
        self.completed_list.setSpacing(6)
        layout.addWidget(self.completed_list)

        self.loadTransformedImages()

        self.setLayout(layout)

    def loadTransformedImages(self):
        """Populate the completed list from pipeline folders already in OUTPUT_DIR."""
        base_dir = str(OUTPUT_DIR)
        paths_list = [os.path.join(base_dir, path) for path in os.listdir(base_dir)]

        for path_i in paths_list:
            if os.path.isdir(path_i):
                params_path = os.path.join(path_i, "params.json")

                if os.path.exists(params_path):
                    folder_name = os.path.basename(path_i)
                    name = folder_name.split('_pipeline')[0]
                    item = QListWidgetItem(f"{name}.png")
                    item.folder_path = path_i

                    img_path = os.path.join(path_i, f"{name}_initial.png")

                    pixmap = QPixmap(img_path)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        icon = QIcon(pixmap)
                        item.setIcon(icon)
                    else:
                        print("Image not found:", img_path)

                    self.completed_list.addItem(item)

    # --- Add a task to the queue ---
    def add_task(self, img_path: str, algo_name: str):
        """Add a new pending-list entry for img_path and return the created item."""
        text = f"{img_path.split('/')[-1]} — {algo_name}"

        # Create the item with an icon on the right
        item = QListWidgetItem(text)
        pixmap = QPixmap(img_path)

        # Shrink the image if it's too large
        if not pixmap.isNull():
            pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon = QIcon(pixmap)
            item.setIcon(icon)

        # Add to the pending list
        self.pending_list.addItem(item)
        return item

    # --- Move the completed item to the right-hand list ---
    def mark_as_done(self, item: QListWidgetItem, folder_path):
        """Move item from the pending list to the completed list."""
        if item is None:
            return
        text = item.text() + " ✅"
        icon = item.icon()
        new_item = QListWidgetItem(icon, text)
        new_item.folder_path = folder_path
        self.completed_list.addItem(new_item)
        row = self.pending_list.row(item)
        self.pending_list.takeItem(row)

    def view_images(self, item: QListWidgetItem):
        """Open an ImageViewer window for the pipeline folder behind item."""
        ImageViewer(item.folder_path)


class ImageViewer(QMainWindow):
    """Window that steps through the saved stages of one dehazing pipeline run."""

    image_pipeline = ["initial", "dc", "tcoarse", "trefined", "final"]

    def __init__(self, dir_path: str):
        super().__init__()
        self.dir_path = dir_path

        # Load parameters from params.json
        json_path = os.path.join(dir_path, "params.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.dehaze_params = data.get("dehaze_params", {})
        self.algo_params = data.get("algo_params", {})

        # --- Central widget ---
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.widget.setMinimumSize(800, 600)
        self.layout_main = QVBoxLayout(self.widget)

        # --- Title at the top ---
        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.layout_main.addWidget(self.title)

        # --- Viewer with vertical buttons ---
        self.visualiser = QWidget()
        self.layout_main.addWidget(self.visualiser)
        self.layout_visual = QHBoxLayout(self.visualiser)

        self.left = QPushButton("final")
        self.right = QPushButton("dc")
        self.left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.right.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Image
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add widgets to the layout
        self.layout_visual.addWidget(self.left)
        self.layout_visual.addWidget(self.label, 1)  # weight = 1 to fill all available space
        self.layout_visual.addWidget(self.right)

        # Connect buttons
        self.left.clicked.connect(lambda: self.change_picture(self.left.text()))
        self.right.clicked.connect(lambda: self.change_picture(self.right.text()))

        # --- Parameters section, side by side ---
        self.params_widget = QWidget()
        self.params_layout = QHBoxLayout(self.params_widget)
        self.layout_main.addWidget(self.params_widget)

        # Dehaze block
        dehaze_block = QVBoxLayout()
        dehaze_block.setAlignment(Qt.AlignTop)
        dehaze_title = QLabel("Dehaze Parameters")
        dehaze_title.setAlignment(Qt.AlignCenter)
        dehaze_title.setStyleSheet("font-weight: bold;")
        dehaze_block.addWidget(dehaze_title)
        self.dehaze_labels = {}
        for k, v in self.dehaze_params.items():
            lbl = QLabel(f"{k} : {v}")
            dehaze_block.addWidget(lbl)
            self.dehaze_labels[k] = lbl

        # Algorithm block
        algo_block = QVBoxLayout()
        algo_block.setAlignment(Qt.AlignTop)
        algo_title = QLabel("Algorithm Parameters")
        algo_title.setAlignment(Qt.AlignCenter)
        algo_title.setStyleSheet("font-weight: bold;")
        algo_block.addWidget(algo_title)
        self.algo_labels = {}
        for k, v in self.algo_params.items():
            lbl = QLabel(f"{k} : {v}")
            algo_block.addWidget(lbl)
            self.algo_labels[k] = lbl

        # Add both blocks side by side
        self.params_layout.addLayout(dehaze_block)
        self.params_layout.addLayout(algo_block)

        # Initial image
        self.current_image_display = ImageViewer.image_pipeline[0]
        self.title.setText(self.current_image_display)

        self.showMaximized()  # open the window maximized
        self.update_image()   # update image once the window has a size

    def resizeEvent(self, event):
        """Keep the displayed image scaled to the window on resize."""
        super().resizeEvent(event)
        self.update_image()

    def update_image(self):
        """Refresh the displayed pixmap for the currently selected pipeline stage."""
        self.current_image_display_path = self.get_image(self.current_image_display)
        pixmap = QPixmap(self.current_image_display_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.label.width(), self.label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(pixmap)

    def change_picture(self, endname):
        """Switch to the pipeline stage named endname and update the left/right buttons."""
        L = len(ImageViewer.image_pipeline)
        i = ImageViewer.image_pipeline.index(endname)
        self.left.setText(ImageViewer.image_pipeline[(i-1) % L])
        self.right.setText(ImageViewer.image_pipeline[(i+1) % L])
        self.current_image_display = ImageViewer.image_pipeline[i]
        self.title.setText(self.current_image_display)
        self.update_image()

    def get_image(self, endname):
        """Return the path of the file in dir_path whose name ends with endname."""
        for filename in os.listdir(self.dir_path):
            full_path = os.path.join(self.dir_path, filename)
            if os.path.isfile(full_path):
                name_without_ext = os.path.splitext(filename)[0]
                if name_without_ext.endswith(endname):
                    return full_path
        return ""


class ImageDropZone(QLabel):
    """Drag-and-drop / click-to-browse area for picking a single source image."""

    def __init__(self, on_image_dropped):
        super().__init__()
        self.on_image_dropped = on_image_dropped
        self.setText("🖼️ Drag or click to choose an image")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #fafafa;
                color: #666;
                font-size: 16px;
            }
            QLabel:hover {
                background-color: #e0e0e0;
            }
        """)
        self.setAcceptDrops(True)
        self.original_pixmap = None
        self.setMinimumSize(400, 300)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.choose_image()
        super().mousePressEvent(event)

    def choose_image(self):
        """Open a dialog to select an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.display_image(file_path)
            self.on_image_dropped(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) for url in urls):
                event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.splitext(file_path)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
                self.display_image(file_path)
                self.on_image_dropped(file_path)
                break

    def display_image(self, path):
        """Load path and show it scaled to the widget's current size."""
        self.original_pixmap = QPixmap(path)
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))


class AlgorithmWidget(QWidget):
    """Algorithm picker, parameter form, run button, and embedded log terminal."""

    ALGO_PARAMS = {
        "chunked_soft_matting": chunk_soft_matting_data.ALGO_PARAMS,
        "soft_matting": soft_matting_data.ALGO_PARAMS,
        "guided_filter": guided_filter_data.ALGO_PARAMS
    }

    DEHAZE_PARAMS = {
        "dc_size": "int",
        "top_percent": "float",
        "patch_avg": "int",
        "omega": "float",
        "t0": "float"
    }

    ALGO_FUNCS = {
        "chunked_soft_matting": chunked_soft_matting,
        "soft_matting": soft_matting,
        "guided_filter": guided_filter
    }

    DEFAULT_DEHAZE_PARAMS = dehazer_data.DEFAULT_DEHAZE_PARAMS

    DEFAULT_ALGO_PARAMS = {
        "chunked_soft_matting": chunk_soft_matting_data.DEFAULT_ALGO_PARAMS,
        "soft_matting": soft_matting_data.DEFAULT_ALGO_PARAMS,
        "guided_filter": guided_filter_data.DEFAULT_ALGO_PARAMS
    }

    def __init__(self):
        super().__init__()
        self.current_algo = None
        self.param_inputs = {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # --- Algorithm selector ---
        self.combo = QComboBox()
        self.combo.addItems(self.ALGO_PARAMS.keys())
        self.combo.currentTextChanged.connect(self.on_algo_changed)
        layout.addWidget(QLabel("Algorithm:"))
        layout.addWidget(self.combo)

        # --- Parameters form ---
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # --- Run button ---
        self.btn_run = QPushButton("Run processing")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #599EFF;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #3E6FB5;
            }
        """)
        layout.addWidget(self.btn_run)

        # --- Embedded terminal ---
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #E3E3E3;
                font-family: Consolas, monospace;
                font-size: 10px;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.terminal.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(QLabel("Terminal:"))
        layout.addWidget(self.terminal, 1)  # takes up the remaining space

        # Initialize with the first algorithm
        self.on_algo_changed(self.combo.currentText())

        # Create the logger
        self.logger = logging.getLogger("widget_logger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():
            handler = QTextEditLogger(self.terminal)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --- Update the form when the algorithm changes ---
    def on_algo_changed(self, algo_name):
        """Rebuild the parameter form for algo_name (general + algorithm-specific fields)."""
        self.current_algo = algo_name

        # Clear the previous form
        while self.form.count():
            item = self.form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_inputs.clear()

        # General dehaze parameters
        for name in self.DEHAZE_PARAMS:
            field = QLineEdit(str(self.DEFAULT_DEHAZE_PARAMS.get(name, "")))
            self.param_inputs[name] = field
            self.form.addRow(name, field)

        # Parameters specific to the chosen algorithm
        for name in self.ALGO_PARAMS[algo_name]:
            field = QLineEdit(str(self.DEFAULT_ALGO_PARAMS[algo_name].get(name, "")))
            self.param_inputs[name] = field
            self.form.addRow(name, field)

        self.log(f"🔄 Parameters loaded for: {algo_name}")

    # --- Retrieve all parameters from the form ---
    def get_current_parameters(self):
        """Read and type-cast the current form values into a dict."""
        params = {}
        for key, field in self.param_inputs.items():
            val_str = field.text().strip()
            if not val_str:
                continue
            typ = self.DEHAZE_PARAMS.get(key, self.ALGO_PARAMS.get(self.current_algo, {}).get(key, "str"))
            if typ == "int":
                params[key] = int(val_str)
            elif typ == "float":
                params[key] = float(val_str)
            else:
                params[key] = val_str
        return params

    def get_selected_algorithm(self):
        """Return the transmission-refinement function for the currently selected algorithm."""
        return self.ALGO_FUNCS[self.current_algo]

    # --- Method to display logs in the terminal ---
    def log(self, message: str):
        """Append a line of text to the embedded terminal."""
        self.terminal.append(message)
        # Move the cursor to the end so the text scrolls
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.terminal.setTextCursor(cursor)


class ImageDifferenceWidget(QWidget):
    """Tab for dropping two images and visualizing their per-channel difference."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Comparator")
        self.setMinimumSize(1300, 800)

        self.image1_path = None
        self.image2_path = None

        # --- Main layout ---
        main_layout = QVBoxLayout(self)

        # --- Drop zones ---
        drop_layout = QHBoxLayout()
        self.zone1 = ImageDropZone(self._on_image1_dropped)
        self.zone2 = ImageDropZone(self._on_image2_dropped)
        drop_layout.addWidget(self.zone1)
        drop_layout.addWidget(self.zone2)
        main_layout.addLayout(drop_layout)

        # --- Compare button ---
        self.btn_compare = QPushButton("Compare the two images")
        self.btn_compare.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_compare.clicked.connect(self.compare_images)
        main_layout.addWidget(self.btn_compare, alignment=Qt.AlignCenter)

        # --- Results display area ---
        self.layout_resultats = QHBoxLayout()
        main_layout.addLayout(self.layout_resultats)
        self.labels_result = []

    # --- Callbacks ---
    def _on_image1_dropped(self, path):
        self.image1_path = path

    def _on_image2_dropped(self, path):
        self.image2_path = path

    # --- Main function ---
    def compare_images(self):
        """Compare the two dropped images and render the diff in the results area."""
        if not self.image1_path or not self.image2_path:
            QMessageBox.warning(self, "Error", "Drop two images before comparing.")
            return

        try:
            # Clear the previously displayed images
            while self.layout_resultats.count():
                item = self.layout_resultats.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            self.labels_result.clear()

            # Compare via the existing utility
            result = ImageComparator.compare_images(self.image1_path, self.image2_path)
            mode = result.get("mode", None)

            if mode == "grayscale" or "diff" in result:
                # Case 1: black & white image
                diff = result["diff"]
                self._display_single_result(diff)

            elif mode == "color" or all(k in result for k in ["diff_r", "diff_g", "diff_b"]):
                # Case 2: color image
                diff_b = result["diff_b"]
                diff_g = result["diff_g"]
                diff_r = result["diff_r"]
                self._display_color_results(diff_b, diff_g, diff_r)

            else:
                QMessageBox.warning(self, "Unknown type", "Comparison format not recognized.")
                return

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

    def _display_single_result(self, diff):
        """Render a single grayscale difference map."""
        block = QVBoxLayout()
        title = QLabel("Difference (Grayscale)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")

        qimg = QImage(
            diff.data, diff.shape[1], diff.shape[0],
            diff.strides[0], QImage.Format_Grayscale8
        )

        lbl_image = ClickableImage(QPixmap.fromImage(qimg))
        lbl_image.setAlignment(Qt.AlignCenter)
        lbl_image.setMinimumSize(300, 300)
        lbl_image.setStyleSheet("border: 1px solid #ccc; background-color: #f8f8f8;")

        block.addWidget(title)
        block.addWidget(lbl_image)
        self.layout_resultats.addLayout(block)
        self.labels_result.append(lbl_image)

    def _display_color_results(self, diff_b, diff_g, diff_r):
        """Render the three per-channel (B/G/R) difference maps side by side."""
        titles = ["Blue Channel", "Green Channel", "Red Channel"]

        for title_text, channel in zip(titles, [diff_b, diff_g, diff_r]):
            block = QVBoxLayout()
            lbl_title = QLabel(title_text)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")

            qimg = QImage(
                channel.data, channel.shape[1], channel.shape[0],
                channel.strides[0], QImage.Format_Grayscale8
            )

            lbl_image = ClickableImage(QPixmap.fromImage(qimg))
            lbl_image.setAlignment(Qt.AlignCenter)
            lbl_image.setMinimumSize(250, 250)
            lbl_image.setStyleSheet("border: 1px solid #ccc; background-color: #f8f8f8;")

            block.addWidget(lbl_title)
            block.addWidget(lbl_image)
            self.layout_resultats.addLayout(block)
            self.labels_result.append(lbl_image)


# --- Pipeline comparison ---

class DirsListSidebar(QListWidget):
    """Draggable sidebar list of pipeline-output folders."""

    def __init__(self):
        super().__init__()
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setStyleSheet("""
            QListWidget {
                background-color: #2E3440;
                color: #ECEFF4;
                border: none;
                padding: 8px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 4px 0;
                border-radius: 6px;
            }
            QListWidget::item:hover {
                background-color: #4C566A;
            }
            QListWidget::item:selected {
                background-color: #5E81AC;
            }
        """)

    def startDrag(self, supportedActions):
        """Start a drag carrying the selected item's folder path as plain text."""
        item = self.currentItem()
        if item:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(item.folder_path)
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.CopyAction)


class ImageComparisonColumn(QWidget):
    """One pipeline stage's images (from two runs) plus their difference, stacked vertically."""

    def __init__(self, img1_path, img2_path, title):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(5)
        label_title = QLabel(title)
        label_title.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setStyleSheet("font-weight: bold; color: #2E3440;")

        for path in [img1_path, img2_path]:
            pixmap = QPixmap(path)
            img_label = ClickableImage(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background: white; border: 1px solid #D8DEE9; border-radius: 8px;")
            layout.addWidget(img_label)

        # Difference
        idiff_dict = ImageComparator.compare_images(img1_path, img2_path)
        if idiff_dict["mode"] == "grayscale":
            diff = idiff_dict["diff"]
            diff = np.ascontiguousarray(np.clip(diff, 0, 255).astype(np.uint8))

            height, width, channels = diff.shape
            if channels != 3:
                raise ValueError("Expected 3 channels for the RGB image.")

            bytesPerLine = diff.strides[0]
            qimg = QImage(diff.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            img_label = ClickableImage(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background: white; border: 1px solid #D8DEE9; border-radius: 8px;")
            layout.addWidget(img_label)

        elif idiff_dict["mode"] == "color":
            placeholder = QLabel("Difference not shown (color image)")
            placeholder.setFixedSize(250, 250)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background: #ECEFF4; border: 1px solid #D8DEE9; border-radius: 8px; color: #4C566A;")
            layout.addWidget(placeholder)

        layout.insertWidget(0, label_title)


class PicturesDropArea(QWidget):
    """Drop target that accepts two pipeline folders (dragged from DirsListSidebar) and compares them."""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.folder_buffer_list = []

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.inner_widget = QWidget()
        self.inner_layout = QHBoxLayout(self.inner_widget)
        self.scroll.setWidget(self.inner_widget)

        self.layout = QVBoxLayout(self)
        self.label = QLabel("💡 Drop two pipelines here to compare them")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #4C566A; font-size: 14px; padding: 20px;")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.scroll)

        self.setStyleSheet("background-color: #ECEFF4; border: 2px dashed #5E81AC; border-radius: 10px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        folder_path = event.mimeData().text()
        self.folder_buffer_list.append(folder_path)
        self.label.setText(f"📂 {len(self.folder_buffer_list)} folder(s) received")
        event.acceptProposedAction()
        if len(self.folder_buffer_list) == 2:
            self.compareTwoImages()

    def clear_inner_layout(self):
        """Remove all widgets and layouts from self.inner_layout."""
        while self.inner_layout.count():
            item = self.inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._delete_layout(item.layout())

    def _delete_layout(self, layout):
        """Recursively remove a layout and its children."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._delete_layout(item.layout())
        del layout

    def compareTwoImages(self):
        """Lay out the two buffered pipeline folders side by side, stage by stage."""
        path1, path2 = self.folder_buffer_list
        self.folder_buffer_list.clear()

        self.clear_inner_layout()

        # Retrieve and sort images
        order = ["initial", "dc", "tcoarse", "trefined", "final"]
        paths1 = sorted(
            [os.path.join(path1, f) for f in os.listdir(path1) if f.endswith(".png")],
            key=lambda x: next((i for i, o in enumerate(order) if o in x), len(order))
        )
        paths2 = sorted(
            [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith(".png")],
            key=lambda x: next((i for i, o in enumerate(order) if o in x), len(order))
        )

        # Load JSON parameters
        with open(os.path.join(path1, "params.json"), "r") as f:
            params1 = json.load(f)
        with open(os.path.join(path2, "params.json"), "r") as f:
            params2 = json.load(f)

        paramsdehaze1 = params1.get("dehaze_params", {})
        paramsalgo1 = params1.get("algo_params", {})
        paramsdehaze2 = params2.get("dehaze_params", {})
        paramsalgo2 = params2.get("algo_params", {})

        # Create the main vertical layout
        main_vlayout = QVBoxLayout()
        main_vlayout.setAlignment(Qt.AlignTop)

        # Widget for pipeline 1
        widget1 = QWidget()
        layout1 = QVBoxLayout(widget1)
        layout1.setAlignment(Qt.AlignTop)

        dehaze_title1 = QLabel("Dehaze Params (Pipeline 1)")
        dehaze_title1.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 2px;")
        layout1.addWidget(dehaze_title1)
        for k, v in paramsdehaze1.items():
            lbl = QLabel(f"{k}: {v}")
            lbl.setStyleSheet("font-size: 11px; margin-left: 4px;")
            layout1.addWidget(lbl)

        algo_title1 = QLabel("Algorithm Params (Pipeline 1)")
        algo_title1.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 4px;")
        layout1.addWidget(algo_title1)
        for k, v in paramsalgo1.items():
            lbl = QLabel(f"{k}: {v}")
            lbl.setStyleSheet("font-size: 11px; margin-left: 4px;")
            layout1.addWidget(lbl)

        # Widget for pipeline 2
        widget2 = QWidget()
        layout2 = QVBoxLayout(widget2)
        layout2.setAlignment(Qt.AlignTop)

        dehaze_title2 = QLabel("Dehaze Params (Pipeline 2)")
        dehaze_title2.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 2px;")
        layout2.addWidget(dehaze_title2)
        for k, v in paramsdehaze2.items():
            lbl = QLabel(f"{k}: {v}")
            lbl.setStyleSheet("font-size: 11px; margin-left: 4px;")
            layout2.addWidget(lbl)

        algo_title2 = QLabel("Algorithm Params (Pipeline 2)")
        algo_title2.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 4px;")
        layout2.addWidget(algo_title2)
        for k, v in paramsalgo2.items():
            lbl = QLabel(f"{k}: {v}")
            lbl.setStyleSheet("font-size: 11px; margin-left: 4px;")
            layout2.addWidget(lbl)

        # Add both widgets to the main vertical layout
        main_vlayout.addWidget(widget1)
        main_vlayout.addSpacing(10)
        main_vlayout.addWidget(widget2)

        # Add the main vertical layout to self.inner_layout
        self.inner_layout.addLayout(main_vlayout)

        for i, stage in enumerate(order):
            if i < len(paths1) and i < len(paths2):
                column = ImageComparisonColumn(paths1[i], paths2[i], stage.upper())
                self.inner_layout.addWidget(column)


class PipelineComparatorWidget(QWidget):
    """Tab that lets the user drag two pipeline-output folders to compare them stage by stage."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pipeline Comparator")
        self.setStyleSheet("font-family: 'Segoe UI';")

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self.sidebar)

        # --- Title + refresh button ---
        title_layout = QHBoxLayout()
        title = QLabel("📁 Pipelines")
        title.setStyleSheet("color: black; font-size: 16px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title)

        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #D8DEE9;
            }
        """)
        refresh_btn.clicked.connect(self.loadTransformedImages)
        title_layout.addWidget(refresh_btn)

        sidebar_layout.addLayout(title_layout)

        # --- Pipeline list ---
        self.sidebar_itemList = DirsListSidebar()
        sidebar_layout.addWidget(self.sidebar_itemList)

        self.pictures_drop_area = PicturesDropArea()

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pictures_drop_area)

        self.loadTransformedImages()

    def loadTransformedImages(self):
        """Refresh the sidebar with every pipeline-output folder found in seriespicturesoutput/."""
        self.sidebar_itemList.clear()
        base_dir = os.path.join(os.getcwd(), "seriespicturesoutput")
        if not os.path.exists(base_dir):
            return

        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            params_path = os.path.join(folder_path, "params.json")
            if not os.path.exists(params_path):
                continue

            name = folder.split('_pipeline')[0]
            item = QListWidgetItem(f"{name}.png")
            item.folder_path = folder_path

            img_path = os.path.join(folder_path, f"{name}_initial.png")
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                icon = QIcon(pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
                item.setIcon(icon)

            self.sidebar_itemList.addItem(item)


class ClickableImage(QLabel):
    """Clickable QLabel displaying a resized image,
    with preview, brightness adjustment, and saving."""

    def __init__(self, pixmap: QPixmap, max_size=250, parent=None):
        super().__init__(parent)
        self.full_pixmap = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            border: 1px solid #D8DEE9;
            border-radius: 8px;
            background: white;
        """)

        if max(pixmap.width(), pixmap.height()) > max_size:
            pixmap = pixmap.scaled(
                max_size, max_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.show_full_image()

    def show_full_image(self):
        """Open a modal dialog with the full-size image, a save button, and a brightness slider."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Image preview")
        dialog.setModal(True)

        main_layout = QHBoxLayout(dialog)

        # Enlarged image
        self.lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.original_pixmap = self.full_pixmap.scaled(
            800, 800,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl.setPixmap(self.original_pixmap)
        main_layout.addWidget(self.lbl)

        # Column for button + slider
        right_layout = QVBoxLayout()
        save_btn = QPushButton("💾", clicked=self.save_current_image)
        right_layout.addWidget(save_btn)

        slider = QSlider(Qt.Orientation.Vertical, minimum=-100, maximum=100, value=0)
        slider.valueChanged.connect(self.adjust_brightness)
        right_layout.addWidget(slider)

        right_layout.addStretch()  # push widgets to the top
        main_layout.addLayout(right_layout)

        dialog.setLayout(main_layout)
        dialog.exec()

    def save_current_image(self):
        """Save the currently displayed (possibly brightness-adjusted) image to disk."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            "",
            "Images (*.png)"
        )
        if file_path:
            self.lbl.pixmap().save(file_path)

    def adjust_brightness(self, value):
        """Re-render the preview image with value added to every RGB channel."""
        qimg = self.original_pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = qimg.width(), qimg.height()

        arr = np.frombuffer(qimg.bits(), dtype=np.uint8).reshape((height, width, 4)).copy()

        rgb = arr[..., :3].astype(np.int16)
        rgb = np.clip(rgb + value, 0, 255).astype(np.uint8)
        arr[..., :3] = rgb

        new_img = QImage(arr.data, width, height, QImage.Format.Format_RGBA8888)
        self.lbl.setPixmap(QPixmap.fromImage(new_img))


class MainWindow(QMainWindow):
    """Top-level window: tabs for processing, queue, image diff, and pipeline diff."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing - Sequential Queue")
        self.resize(1100, 700)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: processing interface
        self.processing_widget = QWidget()
        processing_layout = QHBoxLayout(self.processing_widget)

        self.image_zone = ImageDropZone(self.on_image_dropped)
        self.widget_algo = AlgorithmWidget()
        self.logger = logging.getLogger("widget_logger")

        processing_layout.addWidget(self.image_zone, 2)
        processing_layout.addWidget(self.widget_algo, 1)
        self.widget_algo.btn_run.clicked.connect(self.add_to_queue)

        # Tab 2: queue
        self.queue_widget = QueueWidget()

        # Tab 3: image difference
        self.widget_difference = ImageDifferenceWidget()

        # Tab 4: pipeline difference
        self.widget_difference_pipeline = PipelineComparatorWidget()

        # Add tabs
        self.tabs.addTab(self.processing_widget, "🧩 Image Processing")
        self.tabs.addTab(self.queue_widget, "📜 Queue")
        self.tabs.addTab(self.widget_difference, "➖ Image Difference")
        self.tabs.addTab(self.widget_difference_pipeline, "🔍 Pipeline Difference")

        # --- Queue management ---
        self.image_path = None
        self.queue = []          # list of (path, algo, params, item)
        self.processing_in_progress = False
        self.lock = threading.Lock()

    # --- Drag-and-drop handling ---
    def on_image_dropped(self, path):
        """Remember the dropped image as the current candidate for processing."""
        self.image_path = path
        self.logger.info(f"Image dropped: {path}")

    # --- Add to the queue ---
    def add_to_queue(self):
        """Queue the current image with the currently configured algorithm/parameters."""
        if not self.image_path:
            self.logger.info("No image dropped.")
            return

        algo = self.widget_algo.get_selected_algorithm()
        params = self.widget_algo.get_current_parameters()

        # Create an item in the queue
        item = self.queue_widget.add_task(self.image_path, algo.__name__)
        self.queue.append((self.image_path, algo, params, item))
        self.logger.info(f"🧩 Added to queue: {self.image_path} ({algo.__name__})")

        # Start processing if none is currently running
        if not self.processing_in_progress:
            self._start_next_task()

    # --- Sequential processing ---
    def _start_next_task(self):
        """Pop the next queued task and run it on a background thread, if any remain."""
        if not self.queue:
            self.logger.info("✅ Queue empty.")
            self.processing_in_progress = False
            return

        self.processing_in_progress = True
        path, algo, params, item = self.queue.pop(0)

        def run():
            self.logger.info(f"🚀 Starting processing: {path}")
            folder_path = None
            try:
                # Extract the specific parameters
                dc_size = params.pop("dc_size", 15)
                top_percent = params.pop("top_percent", 0.001)
                patch_avg = params.pop("patch_avg", 2)
                omega = params.pop("omega", 0.95)
                t0 = params.pop("t0", 0.01)

                # Run processing
                folder_path = dehaze(
                    img_path=path,
                    smoothing_method=algo,
                    dc_size=dc_size,
                    top_percent=top_percent,
                    patch_avg=patch_avg,
                    omega=omega,
                    t0=t0,
                    out_dir="seriespicturesoutput",
                    show_steps=True,
                    kwargs=params
                )
            except Exception as e:
                self.logger.warning(f"❌ Error during processing: {e}")

            finally:
                if folder_path:
                    self.queue_widget.mark_as_done(item, folder_path)

                self.logger.info(f"✅ Done: {path}")
                self.processing_in_progress = False
                self._start_next_task()  # automatically starts the next one

        threading.Thread(target=run, daemon=True).start()


class QTextEditLogger(logging.Handler):
    """Logging handler that appends formatted records to a QTextEdit."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        """Append the formatted record and scroll the text edit to the end."""
        msg = self.format(record)
        self.text_edit.append(msg)
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_edit.setTextCursor(cursor)


def main():
    """Launch the Dehazer GUI (used as the ``python -m dehazer`` entry point)."""
    ensure_output_dir()
    app = QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()


if __name__ == "__main__":
    main()
