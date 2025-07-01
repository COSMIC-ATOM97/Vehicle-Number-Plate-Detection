"""
Main ANPR script using TrOCR instead of checkpoint-based OCR
"""

from anpr_trocr_pipeline import TrOCRANPRPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='ANPR with TrOCR')
    parser.add_argument('--yolo_model', required=True, help='Path to YOLO model')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--trocr_model', default='microsoft/trocr-base-printed', 
                       help='TrOCR model (base-printed, large-printed, base-handwritten)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = TrOCRANPRPipeline(args.yolo_model, args.trocr_model)
    results = pipeline.process_image(args.input, save_results=True, confidence_threshold=args.confidence)

if __name__ == '__main__':
    main()
