# WpfAiRunner

A high-performance WPF application for running local AI models via **ONNX Runtime**.
This project demonstrates a production-ready implementation of **LaMa (Inpainting)**, **Depth Anything V2 (Depth Estimation)**, and **Segment Anything (MobileSAM & SAM 2)** with hybrid CPU/GPU execution support.

## ✨ Key Features

### Architecture & Performance
- **Modular Multi-Model UI**: Features a sidebar menu to switch between different model views (`LamaView`, `DepthView`, `SamView`) dynamically.
- **Hybrid Execution**: Supports both **CPU** and **GPU (CUDA)** with a run-time toggle switch.
- **Smart Fallback**: Automatically falls back to CPU if GPU initialization fails.
- **Optimization**: Includes **GPU Warm-up** logic to eliminate initial inference latency and async processing to prevent UI freezing.

### 1. LaMa (Inpainting)
- **Smart Preprocessing**: Automatically crops and resizes the ROI to `512x512`, then pastes the result back to the original resolution.
- **Masking Tools**:
  - **Rect**: Drag to create rectangular masks.
  - **Brush**: Freehand masking with adjustable brush size.

### 2. Depth Anything V2 (Depth Estimation)
- **Visualisation**: Converts model output (disparity) into a normalized grayscale depth map (Brighter = Near, Darker = Far).
- **Fast Inference**: Supports the **V2 Small** model for real-time performance.

### 3. Segment Anything (SAM & SAM 2)
- **Unified Interface**: Supports both **MobileSAM** and **SAM 2** models within a single view.
- **Workflow**:
  1. **Image Encoding**: When an image is opened, the **Encoder** runs immediately to generate embeddings (resized to 1024px).
  2. **Point Prompting**: Clicking any object in the viewer sends the coordinate prompts to the engine.
  3. **Real-time Decoding**: The **Decoder** uses the pre-calculated embeddings and coordinates to generate a segmentation mask.
  4. **Mask Overlay**: The generated mask is dynamically cropped and resized to match the original image resolution.
- **Automated Encoder-Decoder Matching**: Logic to detect and match Encoder/Decoder pairs based on filenames (e.g., matching `tiny` encoder with `tiny` decoder for SAM 2).
- **Mask Post-processing**: Applies **Bicubic Interpolation** and **Soft Masking** (Sigmoid) when upscaling the raw model output (256x256) to the original image size.

## 🛠️ Build & Run

### Prerequisites
- **Visual Studio 2022**
- **.NET 8 SDK**
- **Platform**: Windows x64

### GPU Requirements (Optional)
To enable CUDA acceleration:
- NVIDIA GPU
- **CUDA Toolkit 11.8**
- **cuDNN 8.x** (compatible with CUDA 11.x)
- *Note: If requirements are not met, the app will safely run in CPU mode.*

### Setup
1. Open `WpfAiRunner.sln` in Visual Studio.
2. Restore NuGet packages.
   - **Important**: This project depends on `Microsoft.ML.OnnxRuntime.Gpu` version **1.15.1**. Do not update to newer versions to ensure compatibility.
3. Set the build platform to **x64**.
4. Build and Run the `WpfAiRunner` project.

## 📂 Project Structure

- **WpfAiRunner** (UI): Handles main window, view switching (`Views/`), and user interaction.
- **LamaEngine** (Library): Logic for LaMa Inpainting (Preprocessing, Inference, Postprocessing).
- **DepthEngine** (Library): Logic for Depth Anything V2 estimation.
- **SamEngine** (Library): Unified logic for both **MobileSAM** and **SAM 2** (Encoder-Decoder pipeline, Coordinate mapping, Dynamic mask processing).

## ⚖️ License & Acknowledgements

This project uses third-party open-source software and pretrained models.

### LaMa (Large Mask Inpainting)
- **Original Paper**: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- **Model Source**: [LaMa-ONNX via HuggingFace](https://huggingface.co/Carve/LaMa-ONNX)
  - **Important**: You **MUST** use `lama_fp32.onnx`. (FP16/Quantized models may cause crash due to input type mismatch).

### Depth Anything V2
- **Original Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- **Official Repository**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Model Source**: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small/tree/main/onnx)
  - *Recommended File*: `model.onnx` (located in the `onnx` folder).

### Segment Anything (SAM)

#### MobileSAM
- **Original Repository**: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- **Model Source**: [Acly/MobileSAM via HuggingFace](https://huggingface.co/Acly/MobileSAM/tree/main)
- **Required Models**:
  - `mobile_sam_image_encoder.onnx`
  - `sam_mask_decoder_multi.onnx`

#### SAM 2 (Segment Anything Model 2)
- **Original Repository**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **Model Source**: [vietanhdev/segment-anything-2-onnx-models](https://huggingface.co/vietanhdev/segment-anything-2-onnx-models)
- **Supported Variants**: `tiny`, `small`, `base_plus`, `large` (Encoder & Decoder pairs required).

### Disclaimer
This project is an independent implementation for testing and educational purposes.