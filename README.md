# LamaWpf (LaMa Inpainter WPF Sample)

WPF에서 LaMa Inpainting ONNX 모델을 로드하고,
이미지에서 마스크 영역을 지정한 뒤 인페인팅 결과를 확인하는 샘플입니다.

## 구성
- LamaEngine
  - `LamaInpainter`: ONNX Runtime으로 LaMa 모델 실행 + ROI 기반 처리
- LamaWpf
  - 모델 선택 / 이미지 선택 / 마스크 드래그 / 실행 UI

## Requirements
- Windows
- Visual Studio 2022 (또는 `dotnet` SDK)
- .NET 8
- ONNX Runtime NuGet: **1.15.1** (현재 프로젝트 기준)

## 사용 방법
1. 실행 후 `Pick Model` 클릭 → LaMa `.onnx` 모델 선택
2. `Open Image` 클릭 → 입력 이미지 선택
3. 입력 이미지 위에서 마우스로 드래그 → 마스크(사각형 영역) 지정
4. `Run` 클릭 → 결과 이미지 확인

## 동작 방식(요약)
- 마스크 영역의 bounding box를 구해 ROI로 잡고 정사각형으로 확장
- ROI를 512x512로 리사이즈하여 모델 입력으로 사용
- 모델 출력(512)을 ROI 크기로 되돌린 후 원본에 합성

## Notes
- 현재 마스크 좌표 변환은 Overlay 좌표=이미지 픽셀 좌표 가정에 가까운 단순 구조입니다.
  - `Image.Stretch` 옵션/레이아웃에 따라 정확 변환이 필요하면 별도 보완하세요.