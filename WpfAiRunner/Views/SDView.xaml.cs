using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OnnxEngines.Utils;
using SdGgufEngine;

namespace WpfAiRunner.Views
{
    public partial class SdView : UserControl, IDisposable
    {
        private SdEngine _engine = new SdEngine();
        private string? _modelPath;

        public SdView()
        {
            InitializeComponent();
            _engine.LogAction = (msg) => Dispatcher.Invoke(() => TxtStatus.Text = msg);

            // 수정: 전달받은 float 진행률을 ProgressBar에 반영
            _engine.ProgressAction = (s, t, p) => Dispatcher.Invoke(() =>
            {
                TxtStatus.Text = $"Step {s}/{t} ({p:P0})";
                PbarLoading.IsIndeterminate = false;
                PbarLoading.Value = p * 100;
            });
        }

        public void Dispose() => _engine.Dispose();

        private async void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            _engine.Initialize();

#if DEBUG
            // 1. 디버그 모드에서 지정된 파일명으로 모델 찾기
            // SD v1.5 Q8_0 모델을 기본으로 찾도록 설정
            string? debugPath = OnnxHelper.FindModelInDebug("stable-diffusion-v1-5-Q8_0.gguf");

            if (debugPath != null)
            {
                // 2. 모델이 발견되면 즉시 로딩 시도
                await ReloadModelInternal(debugPath);
            }
            else
            {
                SetStatus("Debug model not found. Please load manually.");
            }
#endif
        }

        // 모델 로딩 로직을 별도 함수로 분리 (BtnPickModel_Click과 공유)
        private async Task ReloadModelInternal(string path)
        {
            _modelPath = path;
            SetBusy(true, "Loading Model...");

            try
            {
                bool useGpu = ChkUseGpu.IsChecked == true;
                // 엔진에서 모델 로드 수행
                bool success = await Task.Run(() => _engine.LoadModel(path, useGpu));

                if (success)
                {
                    TxtModel.Text = Path.GetFileName(path);
                    BtnRun.IsEnabled = true;
                    SetStatus($"Model Loaded on {(useGpu ? "GPU" : "CPU")}.");
                }
                else
                {
                    MessageBox.Show("Failed to load model. Check backend or GPU compatibility.");
                    SetStatus("Load failed.");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}");
            }
            finally
            {
                SetBusy(false);
            }
        }

        // 기존 BtnPickModel_Click은 분리된 함수를 호출하도록 수정
        private async void BtnPickModel_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "GGUF (*.gguf)|*.gguf" };
            if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;

            await ReloadModelInternal(dlg.FileName);
        }

        private async void BtnRun_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_modelPath)) return;

            SetBusy(true, "Generating...");
            TxtPlaceholder.Visibility = Visibility.Collapsed;
            PbarLoading.Value = 0;

            var param = new ModelParameter
            {
                Prompt = TxtPrompt.Text,
                NegativePrompt = TxtNegative.Text,
                Steps = int.Parse(TxtSteps.Text),
                Cfg = float.Parse(TxtCfg.Text),
                Guidance = 1.0f,
                Seed = long.Parse(TxtSeed.Text)
            };

            try
            {
                byte[]? result = await Task.Run(() => _engine.Generate(param));
                if (result != null) ImgOutput.Source = BytesToBitmap(result);
            }
            catch (Exception ex) { MessageBox.Show(ex.Message); }
            finally { SetBusy(false, "Done."); }
        }

        private void SetBusy(bool busy, string? status = null)
        {
            PbarLoading.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
            if (busy) PbarLoading.IsIndeterminate = true;
            BtnRun.IsEnabled = !busy && _modelPath != null;
            if (status != null) TxtStatus.Text = status;
        }

        private void SetStatus(string t) => TxtStatus.Text = t;

        private BitmapImage BytesToBitmap(byte[] b)
        {
            var i = new BitmapImage();
            using var m = new MemoryStream(b);
            i.BeginInit(); i.CacheOption = BitmapCacheOption.OnLoad; i.StreamSource = m; i.EndInit(); i.Freeze();
            return i;
        }
    }
}