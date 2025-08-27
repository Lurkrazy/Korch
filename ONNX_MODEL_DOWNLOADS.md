# ONNX Model Downloads for Korch

This document provides information about where to download ONNX format models that can be used with the Korch framework.

## Overview

Korch is designed to work with ONNX graphs as input. While most benchmark models in the paper are created programmatically from PyTorch models using the `cases/onnx-export/export.py` script, there are also ways to download pre-trained ONNX models from various sources.

## Existing Model Download in Korch

### Candy Style Transfer Model

The repository already includes automatic download functionality for one ONNX model:

**Location**: `cases/onnx-export/export.py` (lines 87-99)

```python
elif args.model == "candy":
    if not args.profile:
        import os
        import onnx_graphsurgeon as gs
        os.system("wget https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx --quiet")
        model = onnx.shape_inference.infer_shapes(simplify(onnx.load("candy-9.onnx"))[0])
        graph = gs.import_onnx(model)
        graph.inputs[0].shape[0] = args.bs
        for node in graph.nodes:
            node.outputs[0].shape[0] = args.bs
        model = gs.export_onnx(graph)
        onnx.save(model, f"candy_bs{args.bs}.onnx")
        os.system("rm candy-9.onnx")
        exit()
```

**Usage**:
```bash
cd cases/onnx-export
python export.py --model candy --bs 1
```

This will download the candy-9.onnx model from the ONNX Model Zoo, modify it for the specified batch size, and save it as `candy_bs1.onnx`.

## ONNX Model Zoo - Primary Source

The main source for pre-trained ONNX models is the official **ONNX Model Zoo**:

### GitHub Repository
- **URL**: https://github.com/onnx/models
- **Direct Model Downloads**: https://github.com/onnx/models/tree/main/validated

### Available Model Categories

1. **Computer Vision**
   - Image Classification (ResNet, VGG, Inception, etc.)
   - Object Detection (YOLO, SSD, etc.)
   - Semantic Segmentation
   - Style Transfer (including the candy model used in Korch)
   - Super Resolution

2. **Natural Language Processing**
   - BERT variants
   - GPT models
   - Translation models

3. **Speech and Audio**
   - Speech recognition models
   - Audio classification models

### Example Downloads from ONNX Model Zoo

You can download models directly using wget or curl:

```bash
# Download candy style transfer model (already implemented in Korch)
wget https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx

# Download ResNet-50 image classification model
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx

# Download YOLO v4 object detection model
wget https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx
```

## Other ONNX Model Sources

### 1. Hugging Face Model Hub
- **URL**: https://huggingface.co/models?library=onnx
- **Filter**: Select "ONNX" under "Libraries" to find ONNX-compatible models
- **Download**: Use `git clone` or download individual files

Example:
```bash
# Clone a model repository
git clone https://huggingface.co/microsoft/DialoGPT-medium

# Or download specific ONNX files using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/DialoGPT-medium', filename='model.onnx')"
```

### 2. NVIDIA NGC Catalog
- **URL**: https://catalog.ngc.nvidia.com/models
- **Filter**: Search for ONNX models
- **Requires**: NVIDIA NGC account for some models

### 3. PyTorch Hub to ONNX Conversion
You can convert PyTorch models from PyTorch Hub to ONNX format:

```python
import torch
import torch.onnx

# Load a model from PyTorch Hub
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True)
```

## Usage with Korch

Once you have downloaded ONNX models, you can use them with Korch:

### 1. Operator Fission
```bash
cd framework
python operator_fission.py path/to/your/model.onnx path/to/output/fissioned_model.onnx
```

### 2. Kernel Orchestration
```bash
python calc.py path/to/your/model.onnx a100 --database_dir model_db
```

## Model Requirements for Korch

When downloading ONNX models for use with Korch, ensure:

1. **ONNX Version Compatibility**: Models should be compatible with the ONNX version installed (check with `pip show onnx`)
2. **Opset Version**: Korch exports models with opset version 11 (see export.py line 228)
3. **Input Shape**: Models should have fixed input shapes or be modifiable using ONNX GraphSurgeon
4. **Hardware Compatibility**: Consider the target hardware (V100, A100, RTX A5000) when selecting models

## Troubleshooting

### Common Issues and Solutions

1. **Large Model Files**: Some ONNX models can be very large (>100MB). Use Git LFS or download directly rather than including in repository.

2. **Shape Inference Issues**: If you encounter shape-related errors, try using ONNX shape inference:
   ```python
   import onnx
   from onnx import shape_inference
   
   model = onnx.load("your_model.onnx")
   inferred_model = shape_inference.infer_shapes(model)
   onnx.save(inferred_model, "your_model_inferred.onnx")
   ```

3. **Batch Size Modification**: Use ONNX GraphSurgeon to modify batch sizes (see candy model example above).

4. **Unsupported Operations**: Some ONNX operations might not be supported by Korch's profilers. Check the operator fission logic in `framework/operator_fission.py` for supported operations.

## Contributing New Model Downloads

If you want to add automatic download functionality for additional models (similar to the candy model), follow this pattern:

1. Add a new model case in `cases/onnx-export/export.py`
2. Include the download URL from a reliable source (preferably ONNX Model Zoo)
3. Add any necessary preprocessing (batch size modification, shape inference, etc.)
4. Test the model with Korch's operator fission and kernel orchestration

Example template:
```python
elif args.model == "your_model_name":
    if not args.profile:
        import os
        import onnx_graphsurgeon as gs
        os.system("wget [MODEL_URL] --quiet")
        model = onnx.shape_inference.infer_shapes(simplify(onnx.load("downloaded_model.onnx"))[0])
        # Add any necessary modifications
        onnx.save(model, f"your_model_bs{args.bs}.onnx")
        os.system("rm downloaded_model.onnx")
        exit()
```

## Resources

- [ONNX Model Zoo](https://github.com/onnx/models)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX GraphSurgeon Documentation](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/)
- [Korch Paper](https://arxiv.org/abs/2406.09465)