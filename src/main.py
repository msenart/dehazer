from dehazer import dehaze
from time import perf_counter
from u_soft_matting_chunked import chunked_soft_matting     # kwargs | maxiter : int, n_cut_width : int, n_cut_height : int, win_radius : int, eps : float, lam : float, max_processes : int, ratio : float
from u_soft_matting import soft_matting                     # kwargs | maxiter : int, win_radius : int, eps : int, lam : int, max_processes : int
from u_guided_filter import guided_filter                   # kwargs | r : int, eps : float
from PySide6.QtWidgets import *
import threading
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize
import os
import json



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
        self.liste_terminees.itemClicked.connect(self.view_images)
        self.liste_terminees.setIconSize(QSize(80, 80))
        self.liste_terminees.setAlternatingRowColors(True)
        self.liste_terminees.setSpacing(6)
        layout.addWidget(self.liste_terminees)

        self.setLayout(layout)

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
        v_image = VisualiseurImage(item.folder_path)

import os
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton, QLabel
from PySide6.QtGui import QPixmap

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

        self.left = QPushButton("dc")
        self.right = QPushButton("final")
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
        "chunked_soft_matting": {
            "maxiter": "int", "n_cut_width": "int", "n_cut_height": "int",
            "win_radius": "int", "eps": "float", "lam": "float",
            "max_processes": "int", "ratio": "float"
        },
        "soft_matting": {
            "maxiter": "int", "win_radius": "int", "eps": "float",
            "lam": "float", "max_processes": "int"
        },
        "guided_filter": {
            "r": "int", "eps": "float"
        }
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

    DEFAULT_DEHAZE_PARAMS = {
    "dc_size": 15,
    "top_percent": 0.001,
    "patch_avg": 2,
    "omega": 0.95,
    "t0": 0.01
    }

    DEFAULT_ALGO_PARAMS = {
        "chunked_soft_matting": {
            "maxiter": 5000,
            "n_cut_width": 1,
            "n_cut_height": 2,
            "win_radius": 3,
            "eps": 1e-7,
            "lam": 1e-4,
            "max_processes": 6,
            "ratio": 0.5
        },
        "soft_matting": {
            "maxiter": 2000,
            "win_radius": 2,
            "eps": 1e-7,
            "lam": 1e-4,
            "max_processes": 6
        },
        "guided_filter": {
            "r": 5,
            "eps": 0.01
        }
    }

    def __init__(self):
        super().__init__()
        self.current_algo = None
        self.param_inputs = {}

        layout = QVBoxLayout(self)

        # Sélecteur d'algorithme
        self.combo = QComboBox()
        self.combo.addItems(self.ALGO_PARAMS.keys())
        self.combo.currentTextChanged.connect(self.on_algo_changed)
        layout.addWidget(QLabel("Algorithme :"))
        layout.addWidget(self.combo)

        # Formulaire de paramètres
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # Bouton de lancement
        self.btn_run = QPushButton("Lancer le traitement")
        layout.addWidget(self.btn_run)

        # Initialiser avec le premier algo
        self.on_algo_changed(self.combo.currentText())

    def on_algo_changed(self, algo_name):
        self.current_algo = algo_name
        # Vider l'ancien formulaire
        while self.form.count():
            item = self.form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_inputs.clear()

            # --- Paramètres de dehaze ---
        for name in self.DEHAZE_PARAMS:
            champ = QLineEdit()
            valeur = str(WidgetAlgorithme.DEFAULT_DEHAZE_PARAMS.get(name, ""))
            champ.setText(valeur)
            self.param_inputs[name] = champ
            self.form.addRow(name, champ)

        # --- Paramètres spécifiques de l'algo ---
        for name in self.ALGO_PARAMS[algo_name]:
            champ = QLineEdit()
            valeur = str(WidgetAlgorithme.DEFAULT_ALGO_PARAMS[algo_name].get(name, ""))
            champ.setText(valeur)
            self.param_inputs[name] = champ
            self.form.addRow(name, champ)

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

        layout_traitement.addWidget(self.zone_image, 2)
        layout_traitement.addWidget(self.widget_algo, 1)
        self.widget_algo.btn_run.clicked.connect(self.ajouter_a_la_file)

        # Onglet 2 : file d’attente
        self.widget_file_attente = WidgetFileAttente()

        # Ajout des onglets
        self.tabs.addTab(self.widget_traitement, "🧩 Traitement d'image")
        self.tabs.addTab(self.widget_file_attente, "📜 File d'attente")

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
            print("✅ File d’attente vide.")
            self.traitement_en_cours = False
            return

        self.traitement_en_cours = True
        path, algo, params, item = self.queue.pop(0)

        def run():
            print(f"🚀 Démarrage du traitement : {path}")
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
                print(f"💾 Paramètres sauvegardés dans {json_path}")

            except Exception as e:
                print(f"❌ Erreur pendant le traitement : {e}")

            finally:
                if folder_path:
                    self.widget_file_attente.marquer_comme_termine(item, folder_path)

                print(f"✅ Terminé : {path}")
                self.traitement_en_cours = False
                self._lancer_prochain_traitement()  # lance le suivant automatiquement

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":

    app = QApplication([])
    mw = FenetrePrincipale()
    mw.show()
    app.exec()

    