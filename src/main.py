from dehazer import dehaze, dehazer_data
from time import perf_counter
from u_soft_matting_chunked import chunked_soft_matting, chunk_soft_matting_data    # kwargs | maxiter : int, n_cut_width : int, n_cut_height : int, win_radius : int, eps : float, lam : float, max_processes : int, ratio : float
from u_soft_matting import soft_matting, soft_matting_data                     # kwargs | maxiter : int, win_radius : int, eps : int, lam : int, max_processes : int
from u_guided_filter import guided_filter, guided_filter_data                  # kwargs | r : int, eps : float
from image_diff import ImageComparator
from PySide6.QtWidgets import *
import threading
from PySide6.QtGui import QPixmap, QIcon, QImage, QTextCursor
from PySide6.QtCore import Qt, QSize
import os
import json
import logging


class WidgetFileAttente(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # Liste de gauche : en attente
        self.liste_attente = QListWidget()
        self.liste_attente.setIconSize(QSize(80, 80))
        self.liste_attente.setAlternatingRowColors(True)
        self.liste_attente.setSpacing(6)
        layout.addWidget(self.liste_attente)

        # Liste de droite : terminés
        self.liste_terminees = QListWidget()
        self.liste_terminees.itemActivated.connect(self.view_images)
        self.liste_terminees.setIconSize(QSize(80, 80))
        self.liste_terminees.setAlternatingRowColors(True)
        self.liste_terminees.setSpacing(6)
        layout.addWidget(self.liste_terminees)

        self.loadTransformedImages()

        self.setLayout(layout)

    def loadTransformedImages(self):
        base_dir = os.path.join(os.getcwd(), "seriespicturesoutput")
        paths_list = [os.path.join(base_dir, path) for path in os.listdir(base_dir)]

        for path_i in paths_list:
            if os.path.isdir(path_i):
                params_path = os.path.join(path_i, "params.json")
                if os.path.exists(params_path):

                    folder_name = os.path.basename(path_i)
                    name = folder_name.split('_')[0]
                    item = QListWidgetItem(f"{name}.png")
                    item.folder_path = path_i

                    img_path = os.path.join(path_i, f"{name}_initial.png")

                    pixmap = QPixmap(img_path)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        icon = QIcon(pixmap)
                        item.setIcon(icon)
                    else:
                        print("Image introuvable :", img_path)

                    self.liste_terminees.addItem(item)

    # ----------------------------------------------------
    # Ajouter un traitement dans la file d’attente
    # ----------------------------------------------------
    def ajouter_traitement(self, img_path: str, algo_name: str):
        texte = f"{img_path.split('/')[-1]} — {algo_name}"

        # Crée l’item avec une icône à droite
        item = QListWidgetItem(texte)
        pixmap = QPixmap(img_path)

        # On réduit l'image si elle est trop grande
        if not pixmap.isNull():
            pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon = QIcon(pixmap)
            item.setIcon(icon)

        # Ajoute dans la liste d’attente
        self.liste_attente.addItem(item)
        return item

    # ----------------------------------------------------
    # Déplace l’item terminé vers la liste de droite
    # ----------------------------------------------------
    def marquer_comme_termine(self, item: QListWidgetItem, folder_path):
        if item is None:
            return
        texte = item.text() + " ✅"
        icon = item.icon()
        new_item = QListWidgetItem(icon, texte)
        new_item.folder_path = folder_path
        self.liste_terminees.addItem(new_item)
        row = self.liste_attente.row(item)
        self.liste_attente.takeItem(row)
    
    def view_images(self, item : QListWidgetItem):
        VisualiseurImage(item.folder_path)

class VisualiseurImage(QMainWindow):
    image_pipeline = ["initial","dc","tcoarse","trefined","final"]

    def __init__(self, dir_path: str):
        super().__init__()
        self.dir_path = dir_path

        # Charger les paramètres depuis params.json
        json_path = os.path.join(dir_path, "params.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.dehaze_params = data.get("dehaze_params", {})
        self.algo_params = data.get("algo_params", {})

        # Widget central
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.widget.setMinimumSize(800, 600)
        self.layout_main = QVBoxLayout(self.widget)

        # Titre en haut
        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.layout_main.addWidget(self.title)

        # Visualiseur avec boutons verticaux
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

        # Ajouter widgets au layout
        self.layout_visual.addWidget(self.left)
        self.layout_visual.addWidget(self.label, 1)  # poids = 1 pour occuper tout l'espace
        self.layout_visual.addWidget(self.right)

        # Connecter boutons
        self.left.clicked.connect(lambda: self.change_picture(self.left.text()))
        self.right.clicked.connect(lambda: self.change_picture(self.right.text()))

        # Section paramètres côte à côte
        self.params_widget = QWidget()
        self.params_layout = QHBoxLayout(self.params_widget)
        self.layout_main.addWidget(self.params_widget)

        # Bloc Dehaze
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

        # Bloc Algo
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

        # Ajouter les deux blocs côte à côte
        self.params_layout.addLayout(dehaze_block)
        self.params_layout.addLayout(algo_block)

        # Initialisation image
        self.current_image_display = VisualiseurImage.image_pipeline[0]
        self.title.setText(self.current_image_display)

        self.showMaximized()  # ouvre la fenêtre maximisée
        self.update_image()   # mettre à jour image après que la fenêtre ait une taille

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()  # redimensionner l'image automatiquement

    def update_image(self):
        self.current_image_display_path = self.get_image(self.current_image_display)
        pixmap = QPixmap(self.current_image_display_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.label.width(), self.label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(pixmap)

    def change_picture(self, endname):
        L = len(VisualiseurImage.image_pipeline)
        i = VisualiseurImage.image_pipeline.index(endname)
        self.left.setText(VisualiseurImage.image_pipeline[(i-1) % L])
        self.right.setText(VisualiseurImage.image_pipeline[(i+1) % L])
        self.current_image_display = VisualiseurImage.image_pipeline[i]
        self.title.setText(self.current_image_display)
        self.update_image()

    def get_image(self, endname):
        for nom_fichier in os.listdir(self.dir_path):
            chemin_complet = os.path.join(self.dir_path, nom_fichier)
            if os.path.isfile(chemin_complet):
                nom_sans_ext = os.path.splitext(nom_fichier)[0]
                if nom_sans_ext.endswith(endname):
                    return chemin_complet
        return ""

class ZoneDepotImage(QLabel):
    def __init__(self, on_image_dropped):
        super().__init__()
        self.on_image_dropped = on_image_dropped
        self.setText("🖼️ Glisse une image ici")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #fafafa;
                color: #666;
                font-size: 16px;
            }
        """)
        self.setAcceptDrops(True)
        self.original_pixmap = None  # stocke le pixmap original
        self.setMinimumSize(400, 300)  # éviter taille trop petite

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')) for url in urls):
                event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            fichier = url.toLocalFile()
            if os.path.splitext(fichier)[1].lower() in ('.png', '.jpg', '.jpeg'):
                self.original_pixmap = QPixmap(fichier)
                self.setPixmap(self.original_pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.on_image_dropped(fichier)
                break

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

class WidgetAlgorithme(QWidget):
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
        
        # --- Sélecteur d'algorithme ---
        self.combo = QComboBox()
        self.combo.addItems(self.ALGO_PARAMS.keys())
        self.combo.currentTextChanged.connect(self.on_algo_changed)
        layout.addWidget(QLabel("Algorithme :"))
        layout.addWidget(self.combo)

        # --- Formulaire des paramètres ---
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # --- Bouton de lancement ---
        self.btn_run = QPushButton("Lancer le traitement")
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

        # --- Terminal intégré ---
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
        layout.addWidget(QLabel("Terminal :"))
        layout.addWidget(self.terminal, 1)  # prend le reste de l’espace

        # Initialiser avec le premier algo
        self.on_algo_changed(self.combo.currentText())

        # Création du logger
        self.logger = logging.getLogger("widget_logger")
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():
            handler = QTextEditLogger(self.terminal)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --- Mise à jour du formulaire quand l’algo change ---
    def on_algo_changed(self, algo_name):
        self.current_algo = algo_name

        # Vider l'ancien formulaire
        while self.form.count():
            item = self.form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_inputs.clear()

        # Paramètres généraux de dehaze
        for name in self.DEHAZE_PARAMS:
            champ = QLineEdit(str(self.DEFAULT_DEHAZE_PARAMS.get(name, "")))
            self.param_inputs[name] = champ
            self.form.addRow(name, champ)

        # Paramètres spécifiques à l’algo choisi
        for name in self.ALGO_PARAMS[algo_name]:
            champ = QLineEdit(str(self.DEFAULT_ALGO_PARAMS[algo_name].get(name, "")))
            self.param_inputs[name] = champ
            self.form.addRow(name, champ)

        self.log(f"🔄 Paramètres chargés pour : {algo_name}")

    # --- Récupérer tous les paramètres du formulaire ---
    def get_current_parameters(self):
        params = {}
        for key, champ in self.param_inputs.items():
            val_str = champ.text().strip()
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
        return self.ALGO_FUNCS[self.current_algo]

    # --- Méthode pour afficher des logs dans le terminal ---
    def log(self, message: str):
        """Ajoute une ligne de texte dans le terminal intégré."""
        self.terminal.append(message)
        # Déplacer le curseur à la fin pour que le texte défile
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.terminal.setTextCursor(cursor)

class WidgetDifference(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comparateur d'images")
        self.setMinimumSize(1300, 800)

        self.image1_path = None
        self.image2_path = None

        # Layout principal
        layout_principal = QVBoxLayout(self)

        # 1️⃣ Zones de dépôt
        layout_drop = QHBoxLayout()
        self.zone1 = ZoneDepotImage(self._on_image1_dropped)
        self.zone2 = ZoneDepotImage(self._on_image2_dropped)
        layout_drop.addWidget(self.zone1)
        layout_drop.addWidget(self.zone2)
        layout_principal.addLayout(layout_drop)

        # 2️⃣ Bouton comparer
        self.btn_compare = QPushButton("Comparer les deux images")
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
        layout_principal.addWidget(self.btn_compare, alignment=Qt.AlignCenter)

        # 3️⃣ Résultats
        layout_resultats = QHBoxLayout()
        titres = ["Canal Bleu", "Canal Vert", "Canal Rouge"]
        self.labels_result = []

        for titre in titres:
            bloc = QVBoxLayout()
            lbl_titre = QLabel(titre)
            lbl_titre.setAlignment(Qt.AlignCenter)
            lbl_titre.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
            lbl_image = QLabel()
            lbl_image.setAlignment(Qt.AlignCenter)
            lbl_image.setMinimumSize(250, 250)
            lbl_image.setStyleSheet("border: 1px solid #ccc; background-color: #f8f8f8;")
            bloc.addWidget(lbl_titre)
            bloc.addWidget(lbl_image)
            layout_resultats.addLayout(bloc)
            self.labels_result.append(lbl_image)

        layout_principal.addLayout(layout_resultats)

    # 🔹 Callbacks
    def _on_image1_dropped(self, path):
        self.image1_path = path

    def _on_image2_dropped(self, path):
        self.image2_path = path

    # 🔹 Fonction principale
    def compare_images(self):
        if not self.image1_path or not self.image2_path:
            QMessageBox.warning(self, "Erreur", "Dépose deux images avant de comparer.")
            return

        try:
            diff_b, diff_g, diff_r = ImageComparator.compare_images(
                self.image1_path, self.image2_path
            )
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            return

        # Convertir en QPixmap et afficher
        for lbl, canal in zip(self.labels_result[:3], [diff_b, diff_g, diff_r]):
            qimg = QImage(canal.data, canal.shape[1], canal.shape[0], canal.strides[0], QImage.Format_Grayscale8)
            lbl.setPixmap(QPixmap.fromImage(qimg).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class FenetrePrincipale(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traitement d'image - File d'attente séquentielle")
        self.resize(1100, 700)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Onglet 1 : interface de traitement
        self.widget_traitement = QWidget()
        layout_traitement = QHBoxLayout(self.widget_traitement)

        self.zone_image = ZoneDepotImage(self.on_image_dropped)
        self.widget_algo = WidgetAlgorithme()
        self.logger = logging.getLogger("widget_logger")

        layout_traitement.addWidget(self.zone_image, 2)
        layout_traitement.addWidget(self.widget_algo, 1)
        self.widget_algo.btn_run.clicked.connect(self.ajouter_a_la_file)

        # Onglet 2 : file d’attente
        self.widget_file_attente = WidgetFileAttente()

        # Onglet 3 : Différence d'images
        self.widget_difference = WidgetDifference()

        # Ajout des onglets
        self.tabs.addTab(self.widget_traitement, "🧩 Traitement d'image")
        self.tabs.addTab(self.widget_file_attente, "📜 File d'attente")
        self.tabs.addTab(self.widget_difference, "➖ Différence d'images")

        # --- Gestion de la file ---
        self.image_path = None
        self.queue = []          # liste de (path, algo, params, item)
        self.traitement_en_cours = False
        self.lock = threading.Lock()

    # ---------------------------------------------------------
    # Gestion du glisser-déposer
    # ---------------------------------------------------------
    def on_image_dropped(self, path):
        self.image_path = path
        print(f"Image déposée : {path}")

    # ---------------------------------------------------------
    # Ajout à la file d’attente
    # ---------------------------------------------------------
    def ajouter_a_la_file(self):
        if not self.image_path:
            print("Aucune image déposée.")
            return

        algo = self.widget_algo.get_selected_algorithm()
        params = self.widget_algo.get_current_parameters()

        # Crée un item dans la file d’attente
        item = self.widget_file_attente.ajouter_traitement(self.image_path, algo.__name__)
        self.queue.append((self.image_path, algo, params, item))
        print(f"🧩 Ajouté à la file : {self.image_path} ({algo.__name__})")

        # Démarre le traitement si aucun n'est en cours
        if not self.traitement_en_cours:
            self._lancer_prochain_traitement()

    # ---------------------------------------------------------
    # Gestion séquentielle
    # ---------------------------------------------------------
    def _lancer_prochain_traitement(self):
        if not self.queue:
            self.logger.info("✅ File d’attente vide.")
            self.traitement_en_cours = False
            return

        self.traitement_en_cours = True
        path, algo, params, item = self.queue.pop(0)

        def run():
            self.logger.info(f"🚀 Démarrage du traitement : {path}")
            folder_path = None
            try:
                # Extraire les paramètres spécifiques
                dc_size = params.pop("dc_size", 15)
                top_percent = params.pop("top_percent", 0.001)
                patch_avg = params.pop("patch_avg", 2)
                omega = params.pop("omega", 0.95)
                t0 = params.pop("t0", 0.01)

                # Lancer le traitement
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

                # --- Écrire params.json dans le dossier de sortie ---
                json_path = os.path.join(folder_path, "params.json")
                params_to_save = {
                    "dehaze_params": {
                        "dc_size": dc_size,
                        "top_percent": top_percent,
                        "patch_avg": patch_avg,
                        "omega": omega,
                        "t0": t0
                    },
                    "algo_params": params
                }
                with open(json_path, "w") as f:
                    json.dump(params_to_save, f, indent=4)
                self.logger.info(f"💾 Paramètres sauvegardés dans {json_path}")

            except Exception as e:
                self.logger.warning(f"❌ Erreur pendant le traitement : {e}")

            finally:
                if folder_path:
                    self.widget_file_attente.marquer_comme_termine(item, folder_path)

                self.logger.info(f"✅ Terminé : {path}")
                self.traitement_en_cours = False
                self._lancer_prochain_traitement()  # lance le suivant automatiquement

        threading.Thread(target=run, daemon=True).start()

class QTextEditLogger(logging.Handler):
    """Handler qui écrit les messages de logging dans un QTextEdit."""
    def __init__(self, text_edit : QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_edit.setTextCursor(cursor)

if __name__ == "__main__":
    app = QApplication([])
    mw = FenetrePrincipale()
    terminal = mw.widget_algo
    mw.show()
    app.exec()

    