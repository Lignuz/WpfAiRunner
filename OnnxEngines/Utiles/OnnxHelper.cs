using Microsoft.ML.OnnxRuntime;

namespace OnnxEngines.Utils;

public static class OnnxHelper
{
    public static (InferenceSession Session, string DeviceMode) LoadSession(string modelPath, bool useGpu)
    {
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC
        };

        string deviceMode = "CPU";

        if (useGpu)
        {
            try
            {
                so.AppendExecutionProvider_CUDA(0);
                deviceMode = "GPU";
            }
            catch
            {
                deviceMode = "CPU (Fallback)";
                System.Diagnostics.Debug.WriteLine("GPU Load Failed. Fallback to CPU.");
            }
        }

        return (new InferenceSession(modelPath, so), deviceMode);
    }
}