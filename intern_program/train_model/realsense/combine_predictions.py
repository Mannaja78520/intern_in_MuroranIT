from PIL import Image, ImageDraw, ImageFont
import os

PREDICT_DIR = "test_predict"
OUTPUT = os.path.join(PREDICT_DIR, "combined_predictions.jpg")

files_labels = [
    ("predict_result_ir_model_ir_image.jpg",   "IR Model  |  IR Image"),
    ("predict_result_ir_model_rgb_image.jpg",  "IR Model  |  RGB Image"),
    ("predict_result_rgb_model_ir_image.jpg",  "RGB Model  |  IR Image"),
    ("predict_result_rgb_model_rgb_image.jpg", "RGB Model  |  RGB Image"),
]

W, H = 640, 480
LABEL_H = 40
PAD = 6

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
except Exception:
    font = ImageFont.load_default()

canvas_w = W * 2 + PAD * 3
canvas_h = (H + LABEL_H) * 2 + PAD * 3
canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
draw = ImageDraw.Draw(canvas)

positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

for (fname, label), (col, row) in zip(files_labels, positions):
    path = os.path.join(PREDICT_DIR, fname)
    img = Image.open(path).resize((W, H))
    x = PAD + col * (W + PAD)
    y = PAD + row * (H + LABEL_H + PAD)
    canvas.paste(img, (x, y + LABEL_H))

    draw.rectangle([x, y, x + W, y + LABEL_H], fill=(20, 20, 20))
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (W - tw) // 2
    ty = y + (LABEL_H - th) // 2
    draw.text((tx, ty), label, fill=(255, 255, 100), font=font)

canvas.save(OUTPUT, quality=95)
print(f"Saved: {OUTPUT}  ({canvas.size[0]}x{canvas.size[1]})")
