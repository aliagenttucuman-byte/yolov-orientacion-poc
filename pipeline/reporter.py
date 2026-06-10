"""
Reporter — genera plots comparativos y CSV de resultados entre versiones YOLO.
Mismo estilo visual que ForestAI.
"""
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


def save_metrics_csv(all_metrics: list[dict], output_path: str):
    """Guarda tabla comparativa de métricas en CSV."""
    if not all_metrics:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"[reporter] Métricas guardadas: {output_path}")


def save_metrics_json(all_metrics: list[dict], output_path: str):
    """Guarda métricas en JSON para consumo programático."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[reporter] JSON guardado: {output_path}")


def plot_comparison(all_metrics: list[dict], output_path: str):
    """
    Bar chart comparando mAP50, Precision y Recall entre modelos.
    Mismo estilo dark/clean que ForestAI.
    """
    if not all_metrics:
        return

    models     = [m["model_key"] for m in all_metrics]
    map50s     = [m.get("mAP50", 0) for m in all_metrics]
    precisions = [m.get("precision", 0) for m in all_metrics]
    recalls    = [m.get("recall", 0) for m in all_metrics]

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars1 = ax.bar(x - width, map50s,     width, label="mAP50",     color="#00d4aa", alpha=0.9)
    bars2 = ax.bar(x,         precisions, width, label="Precision",  color="#0099ff", alpha=0.9)
    bars3 = ax.bar(x + width, recalls,    width, label="Recall",     color="#ff6b6b", alpha=0.9)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")

    ax.set_xlabel("Modelo YOLO", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Comparativa YOLOv — ForestAI Tree Detection", color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.yaxis.grid(True, color="#334155", alpha=0.5)
    ax.legend(facecolor="#1a1a2e", edgecolor="#334155", labelcolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[reporter] Plot guardado: {output_path}")


def annotate_tile(img_path: str, detections: list[dict], output_path: str,
                  model_label: str = "", color: tuple = (0, 212, 170)):
    """
    Dibuja bboxes sobre un tile y guarda la imagen anotada.
    Mismo estilo que ForestAI._draw_results_bbox.
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    for d in detections:
        x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        conf = d.get("confidence", 0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{conf:.2f}", (x1, max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if model_label:
        cv2.putText(img, model_label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
