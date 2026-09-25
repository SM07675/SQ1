import numpy as np
from pathlib import Path
from PIL import Image
from app.services.spectral import extract_water_grounding

def test_urban():
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    water_base = np.array([20, 65, 88], dtype=np.float32)
    noise = np.random.normal(0, 1.5, (h, 96, 3))
    water_pixels = np.clip(water_base + noise, 0, 255).astype(np.uint8)
    img[:, :96] = water_pixels
    img[:, 96:] = [180, 175, 165]
    img[60:70, 96:] = [35, 38, 42]
    img[180:190, 96:] = [30, 32, 36]
    img[:, 170:180] = [28, 30, 35]
    for r in range(10, 240, 30):
        for c in range(105, 245, 35):
            roof_color = [30, 45, 55] if (r + c) % 2 == 0 else [45, 45, 50]
            img[r:r+20, c:c+22] = roof_color
            img[r+20:r+24, c:c+22] = [12, 14, 18]
    scene_path = Path("urban_coastal_tmp.png")
    Image.fromarray(img).save(scene_path)
    res = extract_water_grounding(scene_path, Path("out_tmp"), query="Highlight the largest water body.")
    print("Detected regions:", res["region_count"])
    mask = np.array(Image.open(res["mask_path"]))
    left = np.sum(mask[:, :96] > 0)
    right = np.sum(mask[:, 96:] > 0)
    print("Left water:", left, "Right urban:", right)

if __name__ == "__main__":
    test_urban()
