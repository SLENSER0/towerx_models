# Обучение Моделей Обнаружения Объектов

Этот проект содержит два скрипта для обучения моделей обнаружения объектов:

1. **YOLO Training Script (`trainyolo.py`)**: Обучение модели YOLO с интеграцией Comet.ml для отслеживания экспериментов.
2. **Transformer-Based Detection Training Script (`traindetr.py`)**: Обучение модели обнаружения объектов на основе трансформеров с использованием библиотеки Hugging Face Transformers.

## Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Использование](#использование)
  - [1. Скрипт Обучения YOLO (`trainyolo.py`)](#1-скрипт-обучения-yolo-trainyolopy)
  - [2. Скрипт Обучения на Основе Трансформеров (`traindetr.py`)](#2-скрипт-обучения-на-основе-трансформеров-traindetrpy)


## Требования

Перед запуском скриптов убедитесь, что у вас установлены следующие библиотеки:

- Python 3.10.12
- PyTorch
- Transformers от Hugging Face
- Albumentations
- Supervision
- TorchMetrics
- Comet.ml (только для `train_yolo.py`)
- Другие зависимости, указанные в разделе [Установка](#установка)

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/SLENSER0/towerx_models
   cd towerx_models
   python3 -m venv venv
   source venv/bin/activate  # Для Linux и macOS
   venv\Scripts\activate     # Для Window
   pip install -r requirements.txt

## Использование
1. **Обучение YOLO:**
Пример запуска скрипта для обучения yolo:
```
python train_yolo.py \
  --comet_api_key YOUR_COMET_API_KEY \
  --comet_project_name yolo_project \
  --model_path yolo11x.pt \
  --data_path data.yaml \
  --epochs 25 \
  --imgsz 1520 \
  --device 0 \
  --optimizer Adam \
  --cos_lr \
  --degrees 0.25 \
  --scale 0.3 \
  --save_period 10 \
  --workers 4 \
  --batch 8
```

2. **Обучение RT-DETR:**
Пример запуска скрипта для обучения RT-DETR:
```
python train_object_detection.py \
  --img_size 1536 \
  --output_dir ./finetune_outputs/ \
  --train_images_directory_path /home/ubuntu/dataset/train/images \
  --train_annotations_path /home/ubuntu/dataset/train/instances_train.json \
  --val_images_directory_path /home/ubuntu/dataset/val/images \
  --val_annotations_path /home/ubuntu/dataset/val/instances_val.json
```