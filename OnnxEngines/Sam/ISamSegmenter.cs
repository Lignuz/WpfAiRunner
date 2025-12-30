using System.Collections.Generic;

namespace SamEngine;

public interface ISamSegmenter : IDisposable
{
    string DeviceMode { get; }

    void LoadModels(string encoderPath, string decoderPath, bool useGpu);

    void EncodeImage(byte[] imageBytes);

    (List<float> Scores, byte[] BestMaskBytes, int BestIndex) Predict(float x, float y);

    byte[] GetMaskImage(int index);
}