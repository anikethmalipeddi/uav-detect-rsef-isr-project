from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    import torch
    import torchaudio
except Exception:  # pragma: no cover
    torch = None
    torchaudio = None


def rms_db(wav: np.ndarray, eps: float = 1e-8) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(wav)) + eps) + eps))


def snr_proxy_db(wav: np.ndarray, sr: int, split_hz: int = 1500, eps: float = 1e-8) -> float:
    spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(wav.shape[0], d=1.0 / sr)
    low = np.mean(spec[freqs < split_hz] ** 2) + eps
    high = np.mean(spec[freqs >= split_hz] ** 2) + eps
    return float(10.0 * np.log10(high / low))


def spectral_flatness(wav: np.ndarray, eps: float = 1e-8) -> float:
    spec = np.abs(np.fft.rfft(wav)) + eps
    gmean = np.exp(np.mean(np.log(spec)))
    amean = np.mean(spec)
    return float(gmean / amean)


def zero_cross_rate(wav: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.signbit(wav)).astype(np.float32))))


class LogMel:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: int = 20,
        f_max: int = 8000,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.use_torchaudio = torchaudio is not None and torch is not None

        if self.use_torchaudio:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=2.0,
            )

    def __call__(self, wav: np.ndarray, sr_in: int) -> np.ndarray:
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        if sr_in != self.sample_rate:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=self.sample_rate)

        if self.use_torchaudio:
            wt = torch.from_numpy(wav.astype(np.float32))[None, :]
            mel = self.mel(wt)
            mel = torch.log10(mel + 1e-6)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            return mel.squeeze(0).cpu().numpy().astype(np.float32)

        mel = librosa.feature.melspectrogram(
            y=wav.astype(np.float32),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )
        mel = np.log10(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.astype(np.float32)


def slice_windows(wav: np.ndarray, sr: int, win_s: float = 2.0, hop_s: float = 0.5) -> list[np.ndarray]:
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if wav.shape[0] < win:
        pad = np.zeros((win - wav.shape[0],), dtype=np.float32)
        return [np.concatenate([wav.astype(np.float32), pad], axis=0)]
    out = []
    for start in range(0, wav.shape[0] - win + 1, hop):
        out.append(wav[start : start + win].astype(np.float32))
    return out


def process_audio_tree(
    input_root: Path,
    output_root: Path,
    sample_rate: int = 16000,
    win_s: float = 2.0,
    hop_s: float = 0.5,
    max_files: int = 0,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    logmel = LogMel(sample_rate=sample_rate)
    audio_files = [p for p in input_root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]
    if max_files > 0:
        audio_files = audio_files[:max_files]
    rows = []
    idx = 0
    for wav_path in tqdm(audio_files, desc="Audio windows"):
        wav, sr = sf.read(wav_path, always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.float32)
        windows = slice_windows(wav, sr=sr, win_s=win_s, hop_s=hop_s)
        for w_id, window in enumerate(windows):
            mel = logmel(window, sr_in=sr)
            q = [
                rms_db(window),
                snr_proxy_db(window, sample_rate),
                spectral_flatness(window),
                zero_cross_rate(window),
            ]
            rel = wav_path.relative_to(input_root)
            stem = rel.with_suffix("").as_posix().replace("/", "_")
            out_path = output_root / f"{stem}_{w_id:05d}.npy"
            np.save(out_path, mel.astype(np.float32))
            rows.append(
                {
                    "audio_window_id": idx,
                    "audio_path": str(wav_path),
                    "window_index": w_id,
                    "npy_path": str(out_path),
                    "q_rms_db": q[0],
                    "q_snr_proxy_db": q[1],
                    "q_spectral_flatness": q[2],
                    "q_zcr": q[3],
                    "duration_sec": float(len(window) / sample_rate),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/audio_windows/manifest.csv"))
    args = parser.parse_args()

    df = process_audio_tree(
        input_root=args.input,
        output_root=args.output,
        sample_rate=args.sample_rate,
        win_s=args.window_sec,
        hop_s=args.hop_sec,
        max_files=args.max_files,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest, index=False)
    print(f"Wrote {len(df)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
