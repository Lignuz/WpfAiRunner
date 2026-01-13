using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace WpfAiRunner.Views
{
    // 모든 AI View가 상속받을 뼈대 클래스
    public abstract class BaseAiView : UserControl, IDisposable
    {
        // 공용 이미지 (탭 간 공유용)
        public static BitmapSource? SharedImage { get; set; }

        // 공통 데이터: 현재 작업 중인 입력 이미지
        protected BitmapSource? _inputBitmap;

        // 자식 뷰들이 반드시 연결해줘야 하는 UI 컨트롤들 (추상 프로퍼티)
        protected abstract Image ControlImgInput { get; }
        protected abstract Image? ControlImgOutput { get; }      // 출력이 없는 뷰도 있으므로 nullable
        protected abstract ProgressBar? ControlPbarLoading { get; }
        protected abstract TextBlock? ControlTxtStatus { get; }

        public BaseAiView()
        {
            // 뷰가 로드될 때 공통 로직 실행 (이미지 자동 로드)
            this.Loaded += BaseAiView_Loaded;
        }

        private void BaseAiView_Loaded(object sender, RoutedEventArgs e)
        {
            // 1. 공용(GlobalState) 이미지가 있는지 확인
            if (BaseAiView.SharedImage != null)
            {
                // 내 것과 다르면 업데이트 (중복 로드 방지)
                if (!object.ReferenceEquals(_inputBitmap, BaseAiView.SharedImage))
                {
                    _inputBitmap = BaseAiView.SharedImage;

                    if (ControlImgInput != null)
                        ControlImgInput.Source = _inputBitmap;

                    Log("Shared image loaded automatically.");

                    // 자식 뷰에게 "이미지 새로 들어왔으니 처리해"라고 알림
                    OnImageLoaded();
                }
            }

            // 2. 자식 뷰의 고유 Loaded 로직 실행
            OnLoaded(e);
        }

        // [Overridable] 자식 뷰가 Loaded 시점에 추가로 할 일이 있으면 재정의
        protected virtual void OnLoaded(RoutedEventArgs e) { }

        // [Overridable] 이미지가 로드된 직후 자식 뷰가 할 일 (예: 마스크 버퍼 초기화, 인코딩 등)
        protected virtual void OnImageLoaded() { }

        // [공통 기능] 이미지 열기 다이얼로그
        protected void OpenImageDialog()
        {
            var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.webp" };
            if (dlg.ShowDialog(Window.GetWindow(this)) == true)
            {
                try
                {
                    var bmp = new BitmapImage();
                    bmp.BeginInit();
                    bmp.UriSource = new Uri(dlg.FileName);
                    bmp.CacheOption = BitmapCacheOption.OnLoad;
                    bmp.EndInit();
                    bmp.Freeze(); // 스레드 간 공유 가능

                    _inputBitmap = bmp;
                    if (ControlImgInput != null)
                        ControlImgInput.Source = _inputBitmap;

                    // ★ 핵심: 공용 데이터에 등록하여 다른 탭과 공유
                    BaseAiView.SharedImage = _inputBitmap;

                    Log("Image Loaded.");
                    OnImageLoaded(); // 자식에게 알림
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to load: {ex.Message}");
                }
            }
        }

        // [공통 기능] 결과 이미지 저장
        protected void SaveOutputImage(string defaultName = "result.png")
        {
            if (ControlImgOutput?.Source is not BitmapSource bmp) return;

            var dlg = new SaveFileDialog { Filter = "PNG Image|*.png", FileName = defaultName };
            if (dlg.ShowDialog(Window.GetWindow(this)) == true)
            {
                try
                {
                    using var stream = new FileStream(dlg.FileName, FileMode.Create);
                    var encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(bmp));
                    encoder.Save(stream);
                    Log("Saved successfully.");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Save failed: {ex.Message}");
                }
            }
        }

        // [공통 기능] 상태 메시지 출력
        protected void Log(string msg)
        {
            if (ControlTxtStatus != null)
                ControlTxtStatus.Text = msg;
        }

        // [공통 기능] 로딩 바(Busy) 제어
        protected void SetBusyState(bool isBusy)
        {
            if (ControlPbarLoading != null)
            {
                ControlPbarLoading.Visibility = isBusy ? Visibility.Visible : Visibility.Collapsed;
                ControlPbarLoading.IsIndeterminate = isBusy;
            }
        }

        // [헬퍼] BitmapSource -> byte[] 변환
        protected byte[] BitmapToBytes(BitmapSource bmp)
        {
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            using var ms = new MemoryStream();
            enc.Save(ms);
            return ms.ToArray();
        }

        // [헬퍼] byte[] -> BitmapImage 변환
        protected BitmapImage BytesToBitmap(byte[] bytes)
        {
            var img = new BitmapImage();
            using var ms = new MemoryStream(bytes);
            img.BeginInit();
            img.CacheOption = BitmapCacheOption.OnLoad;
            img.StreamSource = ms;
            img.EndInit();
            img.Freeze();
            return img;
        }

        // 자식 클래스는 반드시 리소스 해제를 구현해야 함
        public abstract void Dispose();
    }
}