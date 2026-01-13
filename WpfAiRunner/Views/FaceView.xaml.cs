using OnnxEngines.Face;
using OnnxEngines.Utils;
using System.Windows;
using System.Windows.Controls;

namespace WpfAiRunner.Views;

public partial class FaceView : BaseAiView
{
    private IFaceDetector? _detector;
    private List<SixLabors.ImageSharp.Rectangle>? _cachedFaces;

    private string? _rfbPath;
    private string? _yolo8Path;
    private string? _yolo11Path;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public FaceView() => InitializeComponent();
    public override void Dispose() => _detector?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        _rfbPath = OnnxHelper.FindModelInDebug("version-RFB-320.onnx");
        _yolo8Path = OnnxHelper.FindModelInDebug("yolov8n-face.onnx");
        _yolo11Path = OnnxHelper.FindModelInDebug("yolov11n-face.onnx");
        await ReloadModel();
#endif
    }

    // 이미지가 바뀌면 캐시 초기화
    protected override void OnImageLoaded()
    {
        _cachedFaces = null;
        ImgOutput.Source = null;
        UpdateUi();
    }

    private async void CboModelSelect_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        SetBusyState(true);
        try
        {
            _detector?.Dispose();
            _detector = null;

            bool useGpu = ChkUseGpu.IsChecked == true;
            int selectedIndex = CboModelSelect.SelectedIndex;
            string modelName = selectedIndex switch { 0 => "RFB-320", 1 => "YOLOv8", 2 => "YOLOv11", _ => "Unknown" };

            await Task.Run(() =>
            {
                if (selectedIndex == 0 && !string.IsNullOrEmpty(_rfbPath))
                    _detector = new FaceDetector(_rfbPath, useGpu);
                else if (selectedIndex == 1 && !string.IsNullOrEmpty(_yolo8Path))
                    _detector = new YoloFaceDetector(_yolo8Path, useGpu);
                else if (selectedIndex == 2 && !string.IsNullOrEmpty(_yolo11Path))
                    _detector = new YoloFaceDetector(_yolo11Path, useGpu);
            });

            if (_detector != null)
            {
                Log($"{modelName} Loaded ({_detector.DeviceMode})");
                if (useGpu && _detector.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;

                // 이미지가 있으면 즉시 재실행
                if (_inputBitmap != null)
                {
                    _cachedFaces = null;
                    await RunDetection();
                }
            }
            else
            {
                Log("Model file not found.");
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load Error: {ex.Message}");
            Log("Load Failed");
        }
        finally
        {
            SetBusyState(false);
            UpdateUi();
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage();

    private async void BtnRun_Click(object sender, RoutedEventArgs e) => await RunDetection();

    private async Task RunDetection()
    {
        if (_detector == null || _inputBitmap == null) return;

        if (_cachedFaces != null)
        {
            await RenderResult();
            return;
        }

        SetBusyState(true);
        Log("Detecting faces...");
        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            _cachedFaces = await Task.Run(() => _detector.DetectFaces(inputBytes));

            if (_cachedFaces.Count == 0)
                Log("No faces found (Original shown).");

            await RenderResult();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Error.");
        }
        finally
        {
            SetBusyState(false);
        }
    }

    private async void ChkOption_Click(object sender, RoutedEventArgs e)
    {
        if (_cachedFaces != null && _cachedFaces.Count > 0)
            await RenderResult();
    }

    private async Task RenderResult()
    {
        if (_detector == null || _inputBitmap == null || _cachedFaces == null) return;

        SetBusyState(true);
        Log("Rendering...");
        try
        {
            bool applyBlur = ChkBlur.IsChecked == true;
            bool drawBox = ChkBox.IsChecked == true;

            // 원본을 매번 다시 바이트로 변환해서 씀 (원본 보존)
            byte[] processedBytes = BitmapToBytes(_inputBitmap);

            if (_cachedFaces.Count > 0)
            {
                await Task.Run(() =>
                {
                    if (applyBlur)
                        processedBytes = _detector.ApplyBlur(processedBytes, _cachedFaces);
                    if (drawBox)
                        processedBytes = _detector.DrawBoundingBoxes(processedBytes, _cachedFaces);
                });
            }
            ImgOutput.Source = BytesToBitmap(processedBytes);

            if (_cachedFaces.Count > 0)
                Log($"Done. {_cachedFaces.Count} faces processed.");
        }
        finally
        {
            SetBusyState(false);
            UpdateUi();
        }
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e) => await ReloadModel();

    private void UpdateUi()
    {
        bool busy = ControlPbarLoading?.Visibility == Visibility.Visible;
        CboModelSelect.IsEnabled = !busy;
        ChkUseGpu.IsEnabled = !busy;
        BtnOpenImage.IsEnabled = !busy && _detector != null;
        BtnRun.IsEnabled = !busy && _detector != null && _inputBitmap != null;
        BtnSave.IsEnabled = !busy && ImgOutput.Source != null;
        ChkBlur.IsEnabled = !busy;
        ChkBox.IsEnabled = !busy;
    }
}