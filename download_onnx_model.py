#!/usr/bin/env python3
"""
Example script demonstrating how to download and prepare ONNX models for Korch.

This script downloads a model from the ONNX Model Zoo and prepares it for use
with Korch's operator fission and kernel orchestration.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

def download_model(url, output_path):
    """Download a model from URL to output_path"""
    print(f"Downloading model from: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path)
        print(f"✓ Downloaded successfully! Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX models for use with Korch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  candy        - Fast neural style transfer (candy style)
  mosaic       - Fast neural style transfer (mosaic style)  
  rain         - Fast neural style transfer (rain princess style)
  resnet50     - ResNet-50 image classification
  mobilenet    - MobileNet v2 image classification
  inception    - Inception v1 image classification
  yolov4       - YOLOv4 object detection
  superres     - Super resolution CNN

Examples:
  %(prog)s candy
  %(prog)s resnet50 --output models/resnet50.onnx
  %(prog)s --list-all
        """
    )
    
    parser.add_argument("model", nargs='?', help="Model to download")
    parser.add_argument("--output", "-o", help="Output path (default: model_name.onnx)")
    parser.add_argument("--list-all", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # Model URLs from ONNX Model Zoo
    models = {
        "candy": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        "mosaic": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "rain": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
        "resnet50": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "mobilenet": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "inception": "https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx",
        "yolov4": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
        "superres": "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
    }
    
    if args.list_all:
        print("Available ONNX models for download:")
        print("=" * 50)
        for name, url in models.items():
            filename = url.split('/')[-1]
            print(f"  {name:12} - {filename}")
        print(f"\nUsage: {sys.argv[0]} <model_name>")
        return
    
    if not args.model:
        parser.print_help()
        return
    
    if args.model not in models:
        print(f"✗ Unknown model: {args.model}")
        print(f"Available models: {', '.join(models.keys())}")
        print(f"Use --list-all to see all available models")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        filename = models[args.model].split('/')[-1]
        output_path = filename
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Download the model
    success = download_model(models[args.model], output_path)
    
    if success:
        print(f"\n✓ Model ready for use with Korch!")
        print(f"\nNext steps:")
        print(f"1. Run operator fission:")
        print(f"   cd framework")
        print(f"   python operator_fission.py {output_path} {output_path.replace('.onnx', '_fissioned.onnx')}")
        print(f"")
        print(f"2. Run kernel orchestration:")
        print(f"   python calc.py {output_path} a100 --database_dir {args.model}_db")
        print(f"")
        print(f"See README.md and ONNX_MODEL_DOWNLOADS.md for more information.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()