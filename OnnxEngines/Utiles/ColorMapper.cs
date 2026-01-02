using SixLabors.ImageSharp.PixelFormats;

namespace OnnxEngines.Utils;

public enum ColormapStyle
{
    Inferno,    // (기본) 보라-주황-노랑 (가장 대중적)
    Magma,      // 검정-보라-분홍-흰색 (대비가 강함)
    Plasma,     // 보라-빨강-노랑 (화려함)
    Viridis,    // 파랑-초록-노랑 (데이터 시각화 표준)
    Grayscale   // 흑백 (기본 깊이맵)
}

public static class ColorMapper
{
    public static Rgba32 GetColor(float value, ColormapStyle style)
    {
        value = Math.Clamp(value, 0.0f, 1.0f);

        if (style == ColormapStyle.Grayscale)
        {
            byte v = (byte)(value * 255);
            return new Rgba32(v, v, v, 255);
        }

        // 각 스타일별 주요 색상 지점 (0.0, 0.25, 0.5, 0.75, 1.0)
        byte[][] palette = style switch
        {
            ColormapStyle.Magma => new byte[][] {
                new byte[] { 0, 0, 4 }, new byte[] { 80, 18, 123 }, new byte[] { 182, 54, 121 }, new byte[] { 251, 135, 97 }, new byte[] { 252, 253, 191 }
            },
            ColormapStyle.Plasma => new byte[][] {
                new byte[] { 13, 8, 135 }, new byte[] { 126, 3, 168 }, new byte[] { 203, 70, 121 }, new byte[] { 248, 149, 64 }, new byte[] { 240, 249, 33 }
            },
            ColormapStyle.Viridis => new byte[][] {
                new byte[] { 68, 1, 84 }, new byte[] { 59, 82, 139 }, new byte[] { 33, 144, 141 }, new byte[] { 93, 201, 99 }, new byte[] { 253, 231, 37 }
            },
            _ => new byte[][] { // Inferno (Default)
                new byte[] { 0, 0, 4 }, new byte[] { 65, 10, 107 }, new byte[] { 187, 55, 84 }, new byte[] { 249, 142, 9 }, new byte[] { 252, 255, 164 }
            }
        };

        // 보간(Interpolation) 로직
        float[] stops = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
        int i = 0;
        for (; i < stops.Length - 2; i++)
        {
            if (value <= stops[i + 1]) break;
        }

        float t = (value - stops[i]) / (stops[i + 1] - stops[i]);

        byte r = (byte)(palette[i][0] + (palette[i + 1][0] - palette[i][0]) * t);
        byte g = (byte)(palette[i][1] + (palette[i + 1][1] - palette[i][1]) * t);
        byte b = (byte)(palette[i][2] + (palette[i + 1][2] - palette[i][2]) * t);

        return new Rgba32(r, g, b, 255);
    }
}