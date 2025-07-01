"""
ANPR Pipeline using TrOCR for text recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
from trocr_ocr import TrOCROCR
import os

class TrOCRANPRPipeline:
    """
    Complete ANPR pipeline with YOLO detection + TrOCR recognition
    """
    
    def __init__(self, yolo_model_path, trocr_model="microsoft/trocr-base-printed"):
        """
        Initialize ANPR pipeline
        
        Args:
            yolo_model_path (str): Path to YOLO model for license plate detection
            trocr_model (str): TrOCR model name for text recognition
        """
        print("🚀 Initializing TrOCR ANPR Pipeline...")
        
        # Load YOLO model for detection
        print("📡 Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load TrOCR model for text recognition
        print("🔤 Loading TrOCR model...")
        self.ocr_model = TrOCROCR(trocr_model)
        
        print("✅ ANPR Pipeline initialized successfully!")
    
    def detect_and_recognize(self, image_path, save_plates=True, confidence_threshold=0.5):
        """
        Detect license plates and recognize text
        
        Args:
            image_path (str): Path to input image
            save_plates (bool): Whether to save extracted plate images
            confidence_threshold (float): Minimum detection confidence
            
        Returns:
            list: List of detected plates with information
        """
        print(f"\n🔍 Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        print(f"📷 Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Run YOLO detection
        print("🎯 Running license plate detection...")
        results = self.yolo_model(image)
        
        plates_info = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                print(f"🎉 Found {len(boxes)} potential license plate(s)")
                
                plate_count = 0
                for j, box in enumerate(boxes):
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detection_confidence = box.conf[0].cpu().numpy()
                    
                    # Filter by confidence threshold
                    if detection_confidence < confidence_threshold:
                        print(f"   ⚠️  Plate {j+1}: Low confidence ({detection_confidence:.3f}), skipping")
                        continue
                    
                    plate_count += 1
                    print(f"\n📋 Processing Plate {plate_count}:")
                    print(f"   📍 Location: [{x1}, {y1}, {x2}, {y2}]")
                    print(f"   🎯 Detection confidence: {detection_confidence:.3f}")
                    
                    # Extract license plate region
                    plate_image = image[y1:y2, x1:x2]
                    
                    if plate_image.size > 0:
                        print(f"   📏 Plate size: {plate_image.shape[1]}x{plate_image.shape[0]} pixels")
                        
                        # Save extracted plate
                        plate_filename = None
                        if save_plates:
                            plate_filename = f"extracted_plate_{timestamp}_{plate_count}.jpg"
                            cv2.imwrite(plate_filename, plate_image)
                            print(f"   💾 Saved extracted plate: {plate_filename}")
                            
                            # Also save a larger version for better viewing
                            large_plate = cv2.resize(plate_image, (400, 200), interpolation=cv2.INTER_CUBIC)
                            large_filename = f"large_plate_{timestamp}_{plate_count}.jpg"
                            cv2.imwrite(large_filename, large_plate)
                            print(f"   🔍 Large version saved: {large_filename}")
                        
                        # Run TrOCR on extracted plate
                        print(f"   🔤 Running TrOCR recognition...")
                        
                        # Try both regular and with confidence
                        recognized_text, ocr_confidence = self.ocr_model.predict_with_confidence(
                            plate_image, return_confidence=True
                        )
                        
                        plate_info = {
                            'plate_number': plate_count,
                            'bbox': [x1, y1, x2, y2],
                            'detection_confidence': float(detection_confidence),
                            'recognized_text': recognized_text,
                            'ocr_confidence': float(ocr_confidence),
                            'plate_image': plate_image,
                            'saved_filename': plate_filename
                        }
                        
                        plates_info.append(plate_info)
                        
                        print(f"   ✅ TrOCR Result: '{recognized_text}' (OCR confidence: {ocr_confidence:.3f})")
                    else:
                        print(f"   ❌ Invalid plate region")
                
                if plate_count == 0:
                    print("⚠️  No plates met the confidence threshold")
            else:
                print("❌ No license plates detected")
        
        return plates_info
    
    def create_visualization(self, image_path, plates_info):
        """
        Create visualization with bounding boxes and recognized text
        
        Args:
            image_path (str): Path to original image
            plates_info (list): List of plate information
            
        Returns:
            numpy.ndarray: Annotated image
        """
        image = cv2.imread(image_path)
        
        for plate_info in plates_info:
            x1, y1, x2, y2 = plate_info['bbox']
            text = plate_info['recognized_text']
            det_conf = plate_info['detection_confidence']
            ocr_conf = plate_info.get('ocr_confidence', 0.0)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Prepare label text
            label = f"{text}"
            conf_label = f"Det:{det_conf:.2f} OCR:{ocr_conf:.2f}"
            
            # Draw text background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            conf_size = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw background rectangles
            cv2.rectangle(image, (x1, y1-40), (x1 + max(label_size[0], conf_size[0]) + 10, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(image, label, (x1+5, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(image, conf_label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def process_image(self, image_path, save_results=True, confidence_threshold=0.5):
        """
        Complete processing pipeline
        
        Args:
            image_path (str): Path to input image
            save_results (bool): Whether to save result files
            confidence_threshold (float): Detection confidence threshold
        """
        # Detect and recognize
        plates_info = self.detect_and_recognize(image_path, save_results, confidence_threshold)
        
        # Create and save visualization
        if plates_info and save_results:
            annotated_image = self.create_visualization(image_path, plates_info)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"trocr_anpr_result_{timestamp}.jpg"
            cv2.imwrite(result_filename, annotated_image)
            print(f"📸 Saved result visualization: {result_filename}")
        
        # Print summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   📷 Input image: {image_path}")
        print(f"   🔍 Total plates detected: {len(plates_info)}")
        
        if plates_info:
            for i, info in enumerate(plates_info, 1):
                print(f"   📋 Plate {i}: '{info['recognized_text']}' ")
                print(f"       🎯 Detection confidence: {info['detection_confidence']:.3f}")
                print(f"       🔤 OCR confidence: {info.get('ocr_confidence', 0.0):.3f}")
        else:
            print("   ❌ No license plates found")
        
        return plates_info

def main():
    parser = argparse.ArgumentParser(description='TrOCR-based ANPR Pipeline')
    parser.add_argument('--yolo_model', required=True, help='Path to YOLO model for detection')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--trocr_model', default='microsoft/trocr-base-printed', 
                       help='TrOCR model name (default: microsoft/trocr-base-printed)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no_save', action='store_true', 
                       help='Do not save extracted plates and results')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = TrOCRANPRPipeline(args.yolo_model, args.trocr_model)
    results = pipeline.process_image(args.input, not args.no_save, args.confidence)

if __name__ == '__main__':
    main()
