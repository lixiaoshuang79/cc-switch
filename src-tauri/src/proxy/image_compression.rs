//! 大图自动压缩：上游网关（如内网 Menshen）通常对请求体有硬限制
//! （nginx `client_max_body_size`，实测 6MB）。Claude Desktop 把图片以
//! base64 全量内嵌在每次请求里，多图长对话很容易顶爆该限制并收到
//! 「Exceeded limit on max bytes to request body」错误。
//!
//! 本模块在本地代理转发前扫描 Anthropic 消息里的 image 块，对体积较大的
//! 图片做缩放 + JPEG 重编码，把请求体压回限制以内。行为克制：
//! - 只处理 > MIN_COMPRESS_BYTES（约 300KB）的图片，小图原样透传；
//! - 最长边缩到 MAX_DIM（1280px），对截图/照片 OCR 与看图足够；
//! - 仅替换 base64 数据，不改动块结构，视觉路由等下游逻辑不受影响。

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::Value;

/// 只有超过该体积（解码后字节数）的图片才压缩。
const MIN_COMPRESS_BYTES: usize = 300 * 1024;
/// 压缩后最长边像素。
const MAX_DIM: u32 = 1280;
/// JPEG 质量。
const JPEG_QUALITY: u8 = 78;

/// 扫描请求体里的 image 块并原地压缩其中的大图。
/// 返回被压缩的图片数量（用于日志）。
pub fn compress_large_images(body: &mut Value) -> usize {
    fn walk(value: &mut Value, compressed: &mut usize) {
        match value {
            Value::Object(map) => {
                // Anthropic image 块：{"type":"image","source":{"type":"base64","data":...}}
                if map.get("type").and_then(Value::as_str) == Some("image") {
                    if let Some(Value::Object(source)) = map.get_mut("source") {
                        if source.get("type").and_then(Value::as_str) == Some("base64") {
                            if let Some(Value::String(data)) = source.get_mut("data") {
                                if let Some(new_data) = compress_base64_if_large(data) {
                                    *data = new_data;
                                    *compressed += 1;
                                }
                            }
                        }
                    }
                }
                for v in map.values_mut() {
                    walk(v, compressed);
                }
            }
            Value::Array(items) => {
                for v in items {
                    walk(v, compressed);
                }
            }
            _ => {}
        }
    }

    let mut compressed = 0;
    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        for m in messages {
            walk(m, &mut compressed);
        }
    }
    compressed
}

/// 若 base64 图片过大则解码、缩放、JPEG 重编码后返回新 base64；否则 None。
fn compress_base64_if_large(data_b64: &str) -> Option<String> {
    let raw = B64.decode(data_b64.trim()).ok()?;
    if raw.len() < MIN_COMPRESS_BYTES {
        return None;
    }
    let img = image::load_from_memory(&raw).ok()?;
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    let resized = if longest > MAX_DIM {
        let scale = MAX_DIM as f32 / longest as f32;
        img.resize(
            (w as f32 * scale).round().max(1.0) as u32,
            (h as f32 * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        img
    };
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    encoder.encode_image(&resized.to_rgb8()).ok()?;
    if buf.len() >= raw.len() {
        // 压缩没有变小（已是最优），保持原数据
        return None;
    }
    Some(B64.encode(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

    fn big_png_base64() -> String {
        // 2000x1500 伪随机噪声图（xorshift），PNG 压不动，体积必然超阈值
        let w = 2000u32;
        let h = 1500u32;
        let mut img = image::RgbImage::new(w, h);
        let mut state = 0x9e3779b97f4a7c15u64;
        for (_, _, p) in img.enumerate_pixels_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *p = image::Rgb([
                (state & 0xFF) as u8,
                ((state >> 8) & 0xFF) as u8,
                ((state >> 16) & 0xFF) as u8,
            ]);
        }
        let mut buf = Vec::new();
        image::codecs::png::PngEncoder::new(&mut buf)
            .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        B64.encode(&buf)
    }

    fn image_message(data: &str) -> Value {
        serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": "看这张图"}
                ]
            }]
        })
    }

    #[test]
    fn compresses_oversized_image_and_keeps_block_structure() {
        let big = big_png_base64();
        let raw_len = B64.decode(&big).unwrap().len();
        assert!(raw_len >= MIN_COMPRESS_BYTES, "测试大图应超过压缩阈值");

        let mut body = image_message(&big);
        let count = compress_large_images(&mut body);
        assert_eq!(count, 1, "应恰好压缩一张大图");

        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], "image", "块结构必须保留");
        assert_eq!(block["source"]["type"], "base64", "source 结构必须保留");
        let new_raw = B64
            .decode(block["source"]["data"].as_str().unwrap())
            .unwrap();
        assert!(
            new_raw.len() * 3 < raw_len,
            "压缩后应显著变小（{} -> {}）",
            raw_len,
            new_raw.len()
        );
        // 重编码产物应是合法 JPEG
        let decoded = image::load_from_memory(&new_raw).unwrap();
        assert!(decoded.width() <= MAX_DIM && decoded.height() <= MAX_DIM);
    }

    #[test]
    fn leaves_small_images_untouched() {
        // 64x64 小图不应被压缩
        let img = image::RgbImage::new(64, 64);
        let mut buf = Vec::new();
        image::codecs::png::PngEncoder::new(&mut buf)
            .write_image(img.as_raw(), 64, 64, image::ExtendedColorType::Rgb8)
            .unwrap();
        let small = B64.encode(&buf);

        let mut body = image_message(&small);
        let count = compress_large_images(&mut body);
        assert_eq!(count, 0);
        assert_eq!(
            body["messages"][0]["content"][0]["source"]["data"],
            small.as_str()
        );
    }

    #[test]
    fn text_only_body_is_unchanged() {
        let mut body = serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "纯文本请求"}]
        });
        let before = body.to_string();
        let count = compress_large_images(&mut body);
        assert_eq!(count, 0);
        assert_eq!(body.to_string(), before);
    }
}
