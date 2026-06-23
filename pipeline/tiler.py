"""
Tiler — mismo patrón que ForestAI.
TIF → tiles JPG con coordenadas georeferenciadas.
"""
import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

TILE_SIZE    = 1024
TILE_OVERLAP = 128
VEG_THRESHOLD = 0.12   # 12% mínimo de píxeles con vegetación (igual que ForestAI)


def tile_has_vegetation(rgb: np.ndarray, threshold: float = VEG_THRESHOLD) -> bool:
    """ExG filter — descartar tiles sin verde. Misma lógica que ForestAI."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    veg_pixels = np.sum(exg > 20)
    total_pixels = rgb.shape[0] * rgb.shape[1]
    return (veg_pixels / total_pixels) >= threshold


def generate_tiles(
    tif_path: str,
    output_dir: str,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    veg_filter: bool = True,
) -> list[dict]:
    """
    Corta el TIF en tiles JPG y devuelve lista de metadatos:
    [{filename, x0, y0, tw, th, crs, bounds}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    step = tile_size - overlap
    tiles = []
    tile_idx = 0

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        crs  = str(src.crs)

        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                tw = min(tile_size, W - x0)
                th = min(tile_size, H - y0)
                if tw < 128 or th < 128:
                    continue

                window = Window(x0, y0, tw, th)
                data   = src.read(list(range(1, min(src.count, 3) + 1)), window=window)
                rgb    = np.transpose(data, (1, 2, 0)).astype(np.uint8)

                if rgb.shape[2] == 1:
                    rgb = np.repeat(rgb, 3, axis=2)

                # Stretch de contraste per-canal: mapea p2-p98 a 0-255
                # Necesario para TIFs subexpuestos (mean ~60 → imagen oscura para VLM)
                rgb_stretched = np.zeros_like(rgb)
                for ch in range(3):
                    p2, p98 = np.percentile(rgb[:, :, ch], (2, 98))
                    if p98 > p2:
                        rgb_stretched[:, :, ch] = np.clip(
                            (rgb[:, :, ch].astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255
                        ).astype(np.uint8)
                    else:
                        rgb_stretched[:, :, ch] = rgb[:, :, ch]
                rgb = rgb_stretched

                if veg_filter and not tile_has_vegetation(rgb):
                    continue

                fname = f"tile_{tile_idx:04d}.jpg"
                fpath = os.path.join(output_dir, fname)
                Image.fromarray(rgb).save(fpath, "JPEG", quality=95)

                # Coordenadas geográficas del tile
                transform  = src.window_transform(window)
                bounds     = rasterio.transform.array_bounds(th, tw, transform)

                tiles.append({
                    "filename": fname,
                    "x0": x0, "y0": y0,
                    "tw": tw, "th": th,
                    "crs": crs,
                    "minX": bounds[0], "minY": bounds[1],
                    "maxX": bounds[2], "maxY": bounds[3],
                })
                tile_idx += 1

    # Guardar CSV de coordenadas (igual que ForestAI/Netflora)
    csv_path = os.path.join(output_dir, "tile_coords.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","x0","y0","tw","th","crs","minX","minY","maxX","maxY"])
        writer.writeheader()
        writer.writerows(tiles)

    print(f"[tiler] {len(tiles)} tiles generados en {output_dir}")
    return tiles
