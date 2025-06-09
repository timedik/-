import sys
sys.path.append(r"C:\Users\SHon\yolov5")  # путь к yolov5

import os
import cv2
import torch
import glob
from pathlib import Path

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

import easyocr  # OCR библиотека

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    coords[:, [0, 2]].clamp_(0, img0_shape[1])
    coords[:, [1, 3]].clamp_(0, img0_shape[0])
    return coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def load_images_from_folder(folder, img_size=640):
    cwd = os.getcwd()
    print(f"[DEBUG] Текущая рабочая папка: {cwd}")
    print(f"[DEBUG] Проверяем папку '{folder}'...")
    if not os.path.exists(folder):
        print(f"[ERROR] Папка '{folder}' не существует!")
        return []
    files = os.listdir(folder)
    print(f"[DEBUG] Найденные файлы: {files}")

    image_paths = glob.glob(os.path.join(folder, '*.*'))
    images = []
    for path in image_paths:
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"[WARNING] Не удалось загрузить изображение: {path}")
            continue
        img, ratio, pad = letterbox(img0, new_shape=(img_size, img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).float() / 255.0
        images.append((path, img.unsqueeze(0), img0, ratio, pad))
    return images

def detect(source='images', weights='yolov5s.pt', imgsz=640, conf_thres=0.25):
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    model.eval()

    reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())

    os.makedirs('outputs', exist_ok=True)
    data = load_images_from_folder(source, img_size=imgsz)

    results = {}

    for path, img, im0s, ratio, pad in data:
        img = img.to(device)
        pred = model(img, augment=False)
        pred = non_max_suppression(pred, conf_thres, 0.45)

        filename = os.path.basename(path)
        results[filename] = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape, (ratio, pad)).round()
                for i, (*xyxy, conf, cls) in enumerate(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    plate = im0s[y1:y2, x1:x2]

                    save_path = f'outputs/{filename}_plate_{i}.jpg'
                    cv2.imwrite(save_path, plate)
                    print(f'[+] Plate saved to {save_path}')

                    results[filename].append([x1, y1, x2, y2])

                    ocr_res = reader.readtext(plate)
                    text = ' '.join([r[1] for r in ocr_res]).strip()
                    with open(f'outputs/{filename}_plate_{i}.txt', 'w', encoding='utf-8') as f:
                        f.write(text)

    return results

if __name__ == '__main__':
    detect()
