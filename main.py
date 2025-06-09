import json
import detect
import recognize

def main():
    print("[🚗] Шаг 1: Обнаружение номеров...")
    boxes = detect.detect()

    print("[🔤] Шаг 2: Распознавание текста...")
    texts = recognize.recognize_text()

    combined_results = []

    for filename, box_list in boxes.items():
        file_plates = []
        for i, box in enumerate(box_list):
            plate_img = f"{filename}_plate_{i}.jpg"
            text = texts.get(plate_img)  # в recognize.py ключи — имена файлов
            if text:
                file_plates.append({
                    "box": box,
                    "text": text
                })

        combined_results.append({
            "filename": filename,
            "plates": file_plates
        })

    with open("final_results.json", "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)

    print("[✓] Результаты сохранены в final_results.json")

if __name__ == '__main__':
    main()
