# LamaWpf

WPF test UI for LaMa (ONNX) inpainting.

## Features
- Load ONNX model (`*.onnx`)
- Load input image
- Create mask with:
  - **Rect**: drag to add multiple rectangles (accumulates)
  - **Brush**: paint with mouse (accumulates), brush size preview circle on hover
- Run inpainting and show output
- **Clear Mask** clears only the mask (keeps output)

## Build / Run
- Visual Studio 2022
- .NET 8 (x64)

Open `LamaWpf.sln` and run `LamaWpf` project.

## Notes
- Mask is stored as **Gray8** (0/255). Multiple regions are supported because the mask is a single bitmap buffer.
- If you want "erase brush" later, you can add a mode to write `0` instead of `255` in `PaintBrushAt()`.
