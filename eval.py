import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Fixed class order (must match training)
CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

AGRAN = {"Lymphocyte", "Monocyte"}
GRAN  = {"Neutrophil", "Eosinophil", "Basophil"}
gran_idx = set(class_to_idx[c] for c in GRAN)

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_file_list(data_dir: str):
    image_paths, labels = [], []
    for c in CLASS_NAMES:
        cdir = os.path.join(data_dir, c)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing class folder: {cdir}")

        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(VALID_EXT)]
        image_paths.extend(files)
        labels.extend([class_to_idx[c]] * len(files))
    return np.array(image_paths), np.array(labels, dtype=np.int32)


def to_binary_np(y):
    yb = np.zeros_like(y)
    yb[np.isin(y, list(gran_idx))] = 1
    return yb


def make_infer_dataset(paths, img_size, batch_size):
    def preprocess(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def save_cm(cm, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print("[INFO] Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Raabin-WBC root folder")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.keras")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    img_size = tuple(args.img_size)

    # Load full dataset (for evaluation you can also load only your test split if you saved it)
    all_images, all_labels = build_file_list(args.data_dir)

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    print("[INFO] Loaded model:", args.model_path)

    # Predict
    ds = make_infer_dataset(all_images, img_size, args.batch_size)
    pred5_prob, pred2_prob = model.predict(ds, verbose=1)

    y_true_5 = all_labels
    y_pred_5 = np.argmax(pred5_prob, axis=1)

    y_true_2 = to_binary_np(y_true_5)
    y_pred_2 = np.argmax(pred2_prob, axis=1)

    # 5-class report
    acc5 = accuracy_score(y_true_5, y_pred_5)
    print("\n[RESULT] 5-class Accuracy:", f"{acc5:.4f}")
    print("\n[REPORT] 5-class Classification Report:\n")
    print(classification_report(y_true_5, y_pred_5, target_names=CLASS_NAMES, digits=4))

    cm5 = confusion_matrix(y_true_5, y_pred_5)
    save_cm(cm5, CLASS_NAMES, "Confusion Matrix (5-class)", os.path.join(fig_dir, "cm_5class.png"))

    # 2-class report
    acc2 = accuracy_score(y_true_2, y_pred_2)
    prec2, rec2, f12, _ = precision_recall_fscore_support(y_true_2, y_pred_2, average="binary", pos_label=1)
    print("\n[RESULT] 2-class (Agran=0 vs Gran=1)")
    print(f"Accuracy        : {acc2:.4f}")
    print(f"Precision (Gran): {prec2:.4f}")
    print(f"Recall (Gran)   : {rec2:.4f}")
    print(f"F1 (Gran)       : {f12:.4f}")

    print("\n[REPORT] 2-class Classification Report:\n")
    print(classification_report(y_true_2, y_pred_2, target_names=["Agranulocytes", "Granulocytes"], digits=4))

    cm2 = confusion_matrix(y_true_2, y_pred_2)
    save_cm(cm2, ["Agran", "Gran"], "Confusion Matrix (2-class)", os.path.join(fig_dir, "cm_2class.png"))

    print("\n[DONE] Figures saved in:", fig_dir)


if __name__ == "__main__":
    main()
