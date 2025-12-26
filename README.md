# WpfAiRunner

A high-performance WPF application for running local AI models via **ONNX Runtime**.
This project demonstrates a production-ready implementation of **LaMa (Large Mask Inpainting)** with hybrid CPU/GPU execution support.

## ✨ Key Features

### Architecture & Performance
- **Modular Design**: UI (`WpfAiRunner`) and Inference Engine (`LamaEngine`) are strictly separated for maintainability.
- **Hybrid Execution**: Supports both **CPU** and **GPU (CUDA)** with a run-time toggle switch.
- **Smart Fallback**: Automatically falls back to CPU if GPU initialization fails (e.g., missing drivers).
- **Optimization**: Includes **Warm-up** logic to eliminate initial inference latency and async processing to prevent UI freezing.

### LaMa Implementation Details
- **Smart Preprocessing**: Automatically crops and resizes the ROI (Region of Interest) to `512x512` for the model, then seamlessly pastes the result back to the original resolution.
- **Auto Scale**: Detects model output range dynamically to ensure correct color rendering.
- **Masking Tools**:
  - **Rect**: Drag to create rectangular masks.
  - **Brush**: Freehand masking with adjustable brush size.

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
   - Core dependency: `Microsoft.ML.OnnxRuntime.Gpu` (v1.15.1).
3. Set the build platform to **x64**.
4. Build and Run the `WpfAiRunner` project.

## 📂 Project Structure

- **WpfAiRunner** (UI): Handles user interaction, rendering, and model selection.
- **LamaEngine** (Library): Encapsulates ONNX session management, tensor processing, and image manipulation logic.

## ⚖️ License & Acknowledgements

This project uses third-party open-source software and pretrained models.

- **Original Paper**: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022)
- **Official Repository**: [advimman/lama](https://github.com/advimman/lama) (Apache 2.0)
- **Model Source**: [LaMa-ONNX via HuggingFace](https://huggingface.co/Carve/LaMa-ONNX)

### Disclaimer
This project is an independent implementation for testing and educational purposes.