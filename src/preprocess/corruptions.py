from __future__ import annotations

import cv2
import numpy as np


def vision_blur(img: np.ndarray, kernel: int) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel, kernel), sigmaX=0)


def vision_low_light(img: np.ndarray, brightness_scale: float, gamma: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * brightness_scale, 0.0, 1.0)
    x = np.power(x, gamma)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def vision_jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out if out is not None else img


def add_awgn_snr(wav: np.ndarray, snr_db: float) -> np.ndarray:
    wav = wav.astype(np.float32)
    sig = np.mean(wav**2) + 1e-8
    noise = sig / (10.0 ** (snr_db / 10.0))
    n = np.random.normal(0.0, np.sqrt(noise), wav.shape).astype(np.float32)
    return wav + n


def hard_clip(wav: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    return np.clip(wav, -threshold, threshold).astype(np.float32)


def wind_like_noise(wav: np.ndarray, strength: float = 0.2) -> np.ndarray:
    n = np.random.normal(0.0, 1.0, wav.shape).astype(np.float32)
    # Simple low-frequency emphasis by cumulative filtering.
    for _ in range(3):
        n = 0.99 * np.concatenate([[0.0], n[:-1]]) + 0.01 * n
    return (wav.astype(np.float32) + strength * n).astype(np.float32)


def radar_clutter_boost(rd: np.ndarray, factor: float) -> np.ndarray:
    rd = rd.astype(np.float32).copy()
    h, w = rd.shape
    c = w // 2
    band = max(1, int(0.08 * w))
    rd[:, c - band : c + band] *= factor
    return rd


def radar_awgn_snr(rd: np.ndarray, snr_db: float) -> np.ndarray:
    rd = rd.astype(np.float32)
    sig = np.mean(rd**2) + 1e-8
    noise = sig / (10.0 ** (snr_db / 10.0))
    return rd + np.random.normal(0.0, np.sqrt(noise), rd.shape).astype(np.float32)
