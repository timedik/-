import os
import pytesseract
import json
import cv2
import re

def preprocess_image(img):
    """Улучшение качества изображения для Tesseract"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def validate_plate_text(text):
    """Проверка валидности номерного знака с улучшенными регулярными выражениями"""
    text = re.sub(r'[\s\-_]+', '', text.strip().upper())
    
    patterns = [
        r'^[АВЕКМНОРСТУХ]{1}\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$',
        r'^[АВЕКМНОРСТУХ]{2}\d{4}$',
        r'^\d{3,4}[АВЕКМНОРСТУХ]{1,2}\d{2,3}$',
        r'^\d{4}[АВЕКМНОРСТУХ]{2}$',
        r'^\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$',
        r'^Т[АВЕКМНОРСТУХ]{1}\d{5}$',
        r'^[АВЕКМНОРСТУХ]{2}\d{3}$',
        r'^[АВЕКМНОРСТУХ]{1}\d{3}[АВЕКМНОРСТУХ]{2}$',
        r'^[A-Z]{1}\d{3}[A-Z]{2}\d{2,3}$',
    ]
    
    return any(re.fullmatch(p, text) for p in patterns)

def normalize_plate_number(text):
    """Приведение номера к стандартному виду"""
    text = re.sub(r'[\s\-_]+', '', text.strip().upper())
    
    if re.fullmatch(r'^[АВЕКМНОРСТУХ]{1}\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$', text):
        return f"{text[:6]} {text[6:]}"
    if re.fullmatch(r'^[АВЕКМНОРСТУХ]{2}\d{4}$', text):
        return f"{text[:2]} {text[2:]}"
    
    return text

def fix_common_ocr_errors(text):
    """Исправление частых ошибок OCR"""
    replacements = {
        '0': 'О',
        '1': 'I',
        '5': 'S',
        '8': 'B',
        '6': 'G',
        '9': 'G',
        '7': 'T',
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

def recognize_text(input_dir='outputs'):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"[!] Папка '{input_dir}' не найдена. Создана новая папка.")

    failed_dir = os.path.join(input_dir, 'failed')
    os.makedirs(failed_dir, exist_ok=True)

    results = []
    total_files = 0
    recognized_count = 0

    char_whitelist = 'АВЕКМНОРСТУХ0123456789'
    configs = [
        f'--psm 7 -c tessedit_char_whitelist={char_whitelist}',
        f'--psm 8 -c tessedit_char_whitelist={char_whitelist}',
        f'--psm 10 -c tessedit_char_whitelist={char_whitelist}',
    ]

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_files += 1
            path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"[!] Не удалось загрузить изображение: {filename}")
                    continue
                
                img_processed = preprocess_image(img)

                recognized_texts = []
                for config in configs:
                    text = pytesseract.image_to_string(
                        img_processed,
                        lang='rus+eng',
                        config=config
                    ).strip()
                    text = fix_common_ocr_errors(text)
                    if text and validate_plate_text(text):
                        recognized_texts.append(text)

                if recognized_texts:
                    best_text = max(recognized_texts, key=len)
                    normalized_text = normalize_plate_number(best_text)
                    results.append({
                        "filename": filename,
                        "text": normalized_text,
                        "raw_text": best_text
                    })
                    recognized_count += 1
                    print(f"[+] {filename}: {normalized_text} (оригинал: {best_text})")
                else:
                    print(f"[-] {filename}: не удалось распознать валидный номер")
                    os.rename(path, os.path.join(failed_dir, filename))

            except Exception as e:
                print(f"[!] Ошибка при обработке {filename}: {str(e)}")

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Обработано файлов: {total_files}")
    print(f"[✓] Успешно распознано: {recognized_count}")
    print(f"[✓] Ошибочных/не распознанных: {total_files - recognized_count}")
    print(f"[✓] Результаты сохранены в results.json")

if __name__ == "__main__":
    test_plates = [
        "А123БЦ116", "АБ1234", "1234АБ", 
        "ТУ12345", "123АБ45", "АБ123", 
        "А123БВ", "A123BC116", "О777ОО177"
    ]

    print("\nТестирование валидации номеров:")
    for plate in test_plates:
        print(f"{plate}: {'✅' if validate_plate_text(plate) else '❌'} → {normalize_plate_number(plate)}")

    recognize_text()
