# 🚗 ANPR-API: Advanced License Plate Recognition System

This project is a high-performance Automatic Number Plate Recognition (ANPR) pipeline that combines YOLOv11 for license plate detection with TrOCR for text recognition.

## 🌟 Features

- 🔍 High-accuracy license plate detection using YOLOv8  
- 🔤 Transformer-based OCR with Microsoft’s TrOCR  
- 📸 Image and video support  
- 🧩 Modular and extensible pipeline  
- ⚡ GPU-accelerated (Colab + local CUDA)  
- 📁 Batch processing for image folders  
- 📊 Dual confidence scores (detection + OCR)

## 📦 Installation

```bash
git clone https://github.com/COSMIC-ATOM97/ANPR-API.git
cd ANPR-API

# Setup virtual environment
python -m venv anpr_env
source anpr_env/bin/activate  # Windows: anpr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 🔹 Single Image Processing

```bash
python run_anpr_trocr.py --yolo_model models/best.pt --input test_images/image.jpg
```

### 🔹 Video Processing (Optional Extension)

```bash
python run_anpr_trocr.py --yolo_model models/best.pt --input input_video.mp4 --output output_video.mp4
```

### 🔹 Batch Processing

```bash
python batch_processor.py --yolo_model models/best.pt --input_dir ./test_images --output_csv results.csv
```

## 🧠 Python API Usage

```python
from anpr_trocr_pipeline import TrOCRANPRPipeline

# Initialize pipeline
anpr = TrOCRANPRPipeline('models/best.pt')

# Run inference
results = anpr.detect_and_recognize('test_images/image.jpg')

for result in results:
    print(f"Detected: {result['recognized_text']}")
    print(f"Detection confidence: {result['detection_confidence']}")
    print(f"OCR confidence: {result['ocr_confidence']}")
    print(f"Bounding box: {result['bbox']}")
```

### 🛠️ Use a Custom TrOCR Model

```python
anpr = TrOCRANPRPipeline(
    'models/best.pt',
    trocr_model='microsoft/trocr-base-handwritten'
)
```

## 🧬 Pipeline Components

- **YOLOv8 Detection**: Detects license plate bounding boxes  
- **Image Preprocessing**: Crops and scales the detected plate region  
- **TrOCR Recognition**: Extracts text from the plate  
- **Postprocessing**: Cleans and validates recognized text  
- **Result Rendering**: Saves annotated images and plate crops

## ⚙️ Command Line Arguments

| Parameter       | Description                          | Default                         |
|----------------|--------------------------------------|---------------------------------|
| `--yolo_model`  | Path to YOLO model file (.pt)        | **Required**                    |
| `--input`       | Path to image or video               | **Required**                    |
| `--trocr_model` | TrOCR model to use                   | `microsoft/trocr-base-printed` |
| `--confidence`  | Detection confidence threshold       | `0.5`                           |
| `--no_save`     | Skip saving output files             | `False`                         |
| `--output`      | (Optional) Output video filename     | `None`                          |

## 📊 Output Format

Each detection returns:

- `recognized_text`: License plate text  
- `detection_confidence`: YOLO prediction score  
- `ocr_confidence`: TrOCR prediction score  
- `bbox`: `[x1, y1, x2, y2]` bounding box  
- `plate_image`: Cropped license plate image  

## 🧪 Performance Tips

- Use high-resolution images when possible  
- Maintain good lighting for better detection  
- Fine-tune TrOCR on your regional license plate data for best results  
- Tweak confidence threshold if detections are missed or too noisy  

## 📁 Model Requirements

- YOLO model trained specifically to detect license plates (single-class preferred)  
- Supported formats: `.pt` (YOLOv5/YOLOv8)  
- TrOCR models downloaded via HuggingFace Transformers  

## 🛠️ Troubleshooting

### CUDA Memory Errors

```bash
export CUDA_VISIBLE_DEVICES=""
python run_anpr_trocr.py ...
```

### Slow Model Load

```bash
# Manually download model in advance
python -c "from transformers import TrOCRProcessor; TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')"
```

## 📈 Roadmap

- [ ] Multi-language OCR (Arabic, Cyrillic, etc.)  
- [ ] Real-time video inference  
- [ ] Dockerized REST API server  
- [ ] Mobile-ready model conversion (e.g., TensorFlow Lite)  
- [ ] Central DB with plate logging and search  
- [ ] Analytics Dashboard (traffic patterns, heatmaps)  

## 🤝 Contributions

Open to:

- 🔧 Feature development  
- 🐞 Bug fixes  
- 📚 Docs and tutorials  
- 🧪 Tests and benchmarks  

Fork → Branch → Commit → Pull Request 🙌

## 📜 License

Licensed under the [MIT License](https://opensource.org/licenses/MIT)

## 🙏 Acknowledgments

- Ultralytics – YOLOv8  
- Microsoft Research – TrOCR  
- Hugging Face – Model hosting  
- PyTorch Team – Core DL framework

## 📞 Support

- 🐛 Report Issues: [GitHub Issues](https://github.com/COSMIC-ATOM97/ANPR-API/issues)  
- 💬 Ask Questions: [GitHub Discussions](https://github.com/COSMIC-ATOM97/ANPR-API/discussions)  
- 🌟 Star this repo if it helps you!

