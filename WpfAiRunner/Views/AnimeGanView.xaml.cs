using OnnxEngines.Style;
using OnnxEngines.Utils;
using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace WpfAiRunner.Views;

public partial class AnimeGanView : BaseAiView
{
    private AnimeGanEngine? _engine;
    private string? _hayaoPath;
    private string? _shinkaiPath;
    private string? _paprikaPath;

    protected override Image ControlImgInput => ImgInput;
    protected override Image? ControlImgOutput => ImgOutput;
    protected override ProgressBar? ControlPbarLoading => PbarLoading;
    protected override TextBlock? ControlTxtStatus => TxtStatus;

    public AnimeGanView() => InitializeComponent();
    public override void Dispose() => _engine?.Dispose();

    protected override async void OnLoaded(RoutedEventArgs e)
    {
#if DEBUG
        _hayaoPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Hayao.onnx");
        _shinkaiPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Shinkai.onnx");
        _paprikaPath = OnnxHelper.FindModelInDebug("AnimeGANv2_Paprika.onnx");
        await ReloadModel();
#endif
        UpdateUi();
    }

    protected override void OnImageLoaded() => UpdateUi();

    private async void CboStyle_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsLoaded) return;
        await ReloadModel();
    }

    private async void ChkUseGpu_Click(object sender, RoutedEventArgs e)
    {
        await ReloadModel();
    }

    private async Task ReloadModel()
    {
        SetBusyState(true);
        try
        {
            bool useGpu = ChkUseGpu.IsChecked == true;
            _engine?.Dispose();
            _engine = new AnimeGanEngine();

            int index = CboStyle.SelectedIndex;
            string? targetPath = index switch
            {
                0 => _hayaoPath,
                1 => _shinkaiPath,
                2 => _paprikaPath,
                _ => _hayaoPath
            };
            string styleName = index switch { 0 => "Hayao", 1 => "Shinkai", 2 => "Paprika", _ => "" };

            if (string.IsNullOrEmpty(targetPath) || !File.Exists(targetPath))
            {
                Log($"{styleName} Model Not Found.");
                BtnRun.IsEnabled = false;
                return;
            }

            await Task.Run(() => _engine.LoadModel(targetPath, useGpu));

            Log($"{styleName} Loaded ({_engine.DeviceMode})");
            if (useGpu && _engine.DeviceMode.Contains("CPU")) ChkUseGpu.IsChecked = false;
            UpdateUi();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Load Failed");
        }
        finally
        {
            SetBusyState(false);
        }
    }

    private void BtnOpenImage_Click(object sender, RoutedEventArgs e) => OpenImageDialog();
    private void BtnSave_Click(object sender, RoutedEventArgs e) => SaveOutputImage("anime_style.png");

    private async void BtnRun_Click(object sender, RoutedEventArgs e)
    {
        if (_engine == null || _inputBitmap == null) return;

        SetBusyState(true);
        try
        {
            byte[] inputBytes = BitmapToBytes(_inputBitmap);
            byte[] resultBytes = await Task.Run(() => _engine.Process(inputBytes));
            ImgOutput.Source = BytesToBitmap(resultBytes);
            Log("Done.");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}");
            Log("Error.");
        }
        finally
        {
            SetBusyState(false);
            UpdateUi();
        }
    }

    private void UpdateUi()
    {
        CboStyle.IsEnabled = ControlPbarLoading?.Visibility != Visibility.Visible;
        BtnOpenImage.IsEnabled = _engine != null;
        BtnRun.IsEnabled = _engine != null && _inputBitmap != null;
        BtnSave.IsEnabled = ImgOutput.Source != null;
    }
}