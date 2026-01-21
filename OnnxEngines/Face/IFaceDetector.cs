using SkiaSharp;

namespace OnnxEngines.Face;

public interface IFaceDetector : IDisposable
{
    string DeviceMode { get; }

    // 얼굴 감지 
    List<SKRectI> DetectFaces(byte[] imageBytes, float confThreshold = 0.5f);

    // 후처리
    byte[] ApplyBlur(byte[] imageBytes, List<SKRectI> faces, int blurSigma = 15);
    byte[] DrawBoundingBoxes(byte[] imageBytes, List<SKRectI> faces, float thickness = 3);
}