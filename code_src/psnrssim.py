import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import complet_dehazer.dehazer as dh
import complet_dehazer.u_guided_filter as gf

# ----------------------------------------------------------------------
# 1) INSÉRER ICI TON ALGO DE DÉBROUILLARDISE
# ----------------------------------------------------------------------
def dehaze_algorithm(image, param):
    kwargs = {
        "r":200, "eps":1e-3
    }
    image = dh.dehaze(image,gf.guided_filter, kwargs,dc_size = param, top_percent = 0.001, patch_avg = 3, omega = 0.95, t0 = 0.01, show_steps = False, custom_output_name=None)
    return image


# ----------------------------------------------------------------------
# 2) PARAMÈTRES
# ----------------------------------------------------------------------
path_hazy = r"C:/Users/mathi/Desktop/Informatique/IMA/dehazer/hazed_images/I-HAZE/train/hazy"
path_gt   = r"C:/Users/mathi/Desktop/Informatique/IMA/dehazer/hazed_images/I-HAZE/train/clear"

parameter_values = [1 + 4*i for i in range(10)]

def to_uint8_image(arr):
    # Si l'image flotante est dans [0,1]
    if arr.max() <= 1.0:
        arr = arr * 255.0

    # Clamp pour éviter les valeurs hors borne
    arr = np.clip(arr, 0, 255)

    # Conversion en uint8
    return arr.astype(np.uint8)

# ----------------------------------------------------------------------
# 3) CALCUL METRIQUES POUR UNE IMAGE
# ----------------------------------------------------------------------
def compute_for_one_image(args):
    hazy_filename, param, hazy_path, gt_path = args

    # Le fichier GT correspondant : on enlève "_hazy"
    clear_filename = hazy_filename.replace("_hazy", "")

    hazy_img_path  = os.path.join(hazy_path, hazy_filename)
    gt_img_path    = os.path.join(gt_path, clear_filename)

    hazy_img = cv2.imread(hazy_img_path)
    gt_img   = cv2.imread(gt_img_path)

    if hazy_img is None:
        print(f"❌ Impossible de lire HAZY : {hazy_img_path}")
        return None

    if gt_img is None:
        print(f"❌ Impossible de lire CLEAR : {gt_img_path}")
        return None

    hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
    gt_img   = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    # ton algo
    dehazed = dehaze_algorithm(hazy_img_path, param)
    dehazed = to_uint8_image(dehazed)

    p = psnr(gt_img, dehazed, data_range=255)
    s = ssim(gt_img, dehazed, data_range=255, channel_axis=2)

    return {
        "image": hazy_filename,
        "param": param,
        "psnr": p,
        "ssim": s
    }



# ----------------------------------------------------------------------
# 4) CALCUL GLOBAL (PSNR et SSIM globaux corrects)
# ----------------------------------------------------------------------
def compute_global_metrics(df):
    global_ssim = df["ssim"].mean()

    mse_list = []
    for _, row in df.iterrows():
        psnr_value = row["psnr"]
        mse = (255 ** 2) / (10 ** (psnr_value / 10))
        mse_list.append(mse)

    mse_global = np.mean(mse_list)
    psnr_global = 10 * np.log10((255 ** 2) / mse_global)

    return psnr_global, global_ssim


# ----------------------------------------------------------------------
# 5) ÉVALUATION MULTIPROCESSING
# ----------------------------------------------------------------------
def evaluate_with_multiprocessing(hazy_dir, gt_dir, params):

    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    gt_files   = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    assert len(hazy_files) == len(gt_files), "Les deux dossiers doivent contenir un nombre identique d'images."

    tasks = []
    for p in params:
        for f in hazy_files:
            tasks.append((f, p, hazy_dir, gt_dir))

    print(f"\n>> Lancement multiprocessing sur {cpu_count()-1} CPU... : nombre de tâches {len(tasks)}")
    with Pool(processes=cpu_count()-1) as pool:
        results = list(tqdm(pool.imap(compute_for_one_image, tasks), total=len(tasks)))

    df = pd.DataFrame(results)
    return df

# ----------------------------------------------------------------------
# 8) GRAPHIQUES
# ----------------------------------------------------------------------
def plot_metric(df, metric_name):
    plt.figure(figsize=(10, 6))
    for image in df["image"].unique():
        sub = df[df["image"] == image]
        plt.plot(sub["param"], sub[metric_name], marker="o", label=image)

    plt.xlabel("Paramètre")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} par image en fonction du paramètre")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_global(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["param"], df["psnr_global"], marker="o")
    plt.xlabel("Paramètre")
    plt.ylabel("PSNR GLOBAL")
    plt.grid(True)
    plt.title("PSNR global en fonction du paramètre")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["param"], df["ssim_global"], marker="o")
    plt.xlabel("Paramètre")
    plt.ylabel("SSIM GLOBAL")
    plt.grid(True)
    plt.title("SSIM global en fonction du paramètre")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    df_results = evaluate_with_multiprocessing(path_hazy, path_gt, parameter_values)
    df_results.to_csv("results_psnr_ssim.csv", index=False)
    print("\n📁 Fichier CSV généré : results_psnr_ssim.csv")

    # --- métriques globales
    global_stats = []
    for param in parameter_values:
        subdf = df_results[df_results["param"] == param]
        psnr_g, ssim_g = compute_global_metrics(subdf)
        global_stats.append({
            "param": param,
            "psnr_global": psnr_g,
            "ssim_global": ssim_g
        })

    df_global = pd.DataFrame(global_stats)
    df_global.to_csv("global_metrics.csv", index=False)
    print("\n📁 Métriques globales : global_metrics.csv")

    print("\n📊 Génération des graphiques...")
    plot_metric(df_results, "psnr")
    plot_metric(df_results, "ssim")
    plot_global(df_global)

    print("\n🎉 Programme terminé avec succès !")