"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Data Preparation Pipeline                             ║
║                                                                      ║
║  Transforms raw call recordings into TTS-ready training data:        ║
║                                                                      ║
║    Step 1: Discover & convert audio files (any format → WAV)         ║
║    Step 2: Speaker diarization (separate agent from customer)        ║
║    Step 3: Audio denoising (remove background noise)                 ║
║    Step 4: Segment extraction (split into utterances)                ║
║    Step 5: Quality filtering (SNR, duration, silence ratio)          ║
║    Step 6: Transcription (Whisper ASR → text)                        ║
║    Step 7: Build metadata.csv (training-ready format)                ║
║                                                                      ║
║  Usage:                                                              ║
║    python scripts/prepare_data.py \                                  ║
║        --input data/01_raw_calls/ \                                  ║
║        --output data/04_processed/ \                                 ║
║        --speaker agent \                                             ║
║        --whisper-url http://gpu-server:8123/transcribe               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("callex.data.prepare")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PrepConfig:
    """Data preparation pipeline configuration."""
    # Audio
    target_sr: int = 24000
    target_channels: int = 1
    target_bit_depth: int = 16
    
    # Segmentation
    min_duration_sec: float = 0.5
    max_duration_sec: float = 15.0
    silence_threshold_db: float = -40.0
    min_silence_duration_sec: float = 0.3
    
    # Quality filtering
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.5     # Max 50% silence in a segment
    min_speech_ratio: float = 0.3      # Min 30% speech in a segment
    
    # Speaker diarization
    speaker_to_keep: str = "SPEAKER_00"  # Which speaker channel to keep
    
    # Transcription
    whisper_model: str = "large-v3"
    whisper_language: str = "hi"
    whisper_url: Optional[str] = None  # Use remote Whisper server
    
    # Output
    file_prefix: str = "CX"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 1: Audio Discovery & Conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus', '.webm'}


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all audio files in directory."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    files.sort()
    logger.info("📁 Discovered %d audio files in %s", len(files), input_dir)
    return files


def convert_to_wav(input_path: Path, output_path: Path, sr: int = 24000) -> bool:
    """
    Convert any audio format to standardized WAV using ffmpeg.
    Output: 24kHz, mono, 16-bit PCM.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(sr),         # Sample rate
            "-ac", "1",             # Mono
            "-sample_fmt", "s16",   # 16-bit
            "-f", "wav",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("❌ ffmpeg failed on %s: %s", input_path.name, e.stderr[-200:] if e.stderr else "")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found! Install: apt install ffmpeg / brew install ffmpeg")
        sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 2: Speaker Diarization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def diarize_audio(wav_path: Path, config: PrepConfig) -> list[dict]:
    """
    Run speaker diarization to identify speaker segments.
    
    Returns list of: {"speaker": str, "start": float, "end": float}
    
    Uses pyannote.audio if available, falls back to simple
    energy-based VAD segmentation.
    """
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN"),
        )
        diarization = pipeline(str(wav_path))
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        
        logger.info("  🎙️ Diarization: %d segments, %d speakers",
                     len(segments), len(set(s["speaker"] for s in segments)))
        return segments
        
    except ImportError:
        logger.info("  ℹ️ pyannote not installed — using VAD-based segmentation")
        return _vad_segmentation(wav_path, config)
    except Exception as e:
        logger.warning("  ⚠️ Diarization failed: %s — falling back to VAD", e)
        return _vad_segmentation(wav_path, config)


def _vad_segmentation(wav_path: Path, config: PrepConfig) -> list[dict]:
    """
    Simple energy-based Voice Activity Detection segmentation.
    Splits audio on silence regions. All segments are labeled
    as the same speaker (assumes single-speaker or pre-separated audio).
    """
    audio = _load_wav_numpy(wav_path)
    sr = config.target_sr
    
    # Frame-level energy
    frame_size = int(0.025 * sr)   # 25ms frames
    hop_size = int(0.010 * sr)     # 10ms hop
    threshold = 10 ** (config.silence_threshold_db / 20.0)
    
    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms > threshold)
    
    # Find speech regions
    segments = []
    in_speech = False
    start = 0
    min_silence_frames = int(config.min_silence_duration_sec / 0.010)
    silence_count = 0
    
    for i, is_speech in enumerate(frames):
        if is_speech:
            if not in_speech:
                start = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end = i - silence_count
                    segments.append({
                        "speaker": config.speaker_to_keep,
                        "start": start * 0.010,
                        "end": end * 0.010,
                    })
                    in_speech = False
    
    # Handle last segment
    if in_speech:
        segments.append({
            "speaker": config.speaker_to_keep,
            "start": start * 0.010,
            "end": len(frames) * 0.010,
        })
    
    logger.info("  🔊 VAD: %d speech segments found", len(segments))
    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 3: Denoising
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def denoise_audio(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Apply noise reduction to audio segment.
    Uses noisereduce if available, otherwise spectral subtraction.
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    except ImportError:
        return _spectral_subtraction(audio, sr)


def _spectral_subtraction(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple spectral subtraction for noise reduction.
    Estimates noise from the first 0.5s (assumed silence/noise).
    """
    n_fft = 2048
    hop = 512
    
    # Estimate noise spectrum from first 0.5s
    noise_samples = audio[:int(0.5 * sr)]
    if len(noise_samples) < n_fft:
        return audio
    
    # Compute noise spectrum
    noise_stft = np.fft.rfft(noise_samples[:n_fft])
    noise_magnitude = np.abs(noise_stft)
    
    # Process in blocks
    output = np.copy(audio)
    for i in range(0, len(audio) - n_fft, hop):
        block = audio[i:i + n_fft]
        spectrum = np.fft.rfft(block)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Subtract noise (with flooring to prevent artifacts)
        clean_magnitude = np.maximum(magnitude - noise_magnitude * 1.5, magnitude * 0.05)
        
        # Reconstruct
        clean_spectrum = clean_magnitude * np.exp(1j * phase)
        clean_block = np.fft.irfft(clean_spectrum)
        
        output[i:i + n_fft] = clean_block
    
    return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 4: Segment Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_segment(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    """Extract audio segment by timestamp."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 5: Quality Filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_quality(audio: np.ndarray, sr: int, config: PrepConfig) -> tuple[bool, str]:
    """
    Check if an audio segment passes quality thresholds.
    
    Returns (passed, reason).
    """
    duration = len(audio) / sr
    
    # Duration check
    if duration < config.min_duration_sec:
        return False, f"too_short ({duration:.1f}s)"
    if duration > config.max_duration_sec:
        return False, f"too_long ({duration:.1f}s)"
    
    # SNR estimation
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    snr_db = 20 * np.log10(rms + 1e-10) + 96  # Approximate SNR relative to noise floor
    if snr_db < config.min_snr_db:
        return False, f"low_snr ({snr_db:.1f} dB)"
    
    # Silence ratio
    threshold = 10 ** (config.silence_threshold_db / 20.0)
    frame_size = int(0.025 * sr)
    n_frames = max(1, len(audio) // frame_size)
    silent_frames = sum(
        1 for i in range(0, len(audio) - frame_size, frame_size)
        if np.sqrt(np.mean(audio[i:i + frame_size] ** 2)) < threshold
    )
    silence_ratio = silent_frames / n_frames
    
    if silence_ratio > config.max_silence_ratio:
        return False, f"too_much_silence ({silence_ratio:.0%})"
    
    speech_ratio = 1.0 - silence_ratio
    if speech_ratio < config.min_speech_ratio:
        return False, f"too_little_speech ({speech_ratio:.0%})"
    
    # Clipping check
    clip_ratio = np.mean(np.abs(audio) > 0.95)
    if clip_ratio > 0.01:
        return False, f"clipping ({clip_ratio:.1%})"
    
    return True, "ok"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 6: Transcription
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def transcribe_segment(wav_path: Path, config: PrepConfig) -> str:
    """
    Transcribe an audio segment to text.
    Uses remote Whisper server if URL is configured, else local Whisper.
    """
    if config.whisper_url:
        return _transcribe_remote(wav_path, config)
    return _transcribe_local(wav_path, config)


def _transcribe_remote(wav_path: Path, config: PrepConfig) -> str:
    """Transcribe using remote Whisper server (your GPU STT)."""
    import requests
    
    try:
        with open(wav_path, 'rb') as f:
            response = requests.post(
                config.whisper_url,
                files={"file": (wav_path.name, f, "audio/wav")},
                data={"language": config.whisper_language},
                timeout=30,
            )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "").strip()
        else:
            logger.warning("Whisper API error %d: %s", response.status_code, response.text[:100])
            return ""
    except Exception as e:
        logger.warning("Whisper API failed: %s", e)
        return ""


def _transcribe_local(wav_path: Path, config: PrepConfig) -> str:
    """Transcribe using local Whisper model."""
    try:
        import whisper
        model = whisper.load_model(config.whisper_model)
        result = model.transcribe(str(wav_path), language=config.whisper_language)
        return result.get("text", "").strip()
    except ImportError:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(wav_path), language=config.whisper_language)
            return " ".join(s.text for s in segments).strip()
        except ImportError:
            logger.error("Neither whisper nor faster-whisper installed!")
            return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_wav_numpy(path: Path) -> np.ndarray:
    """Load WAV file as float32 numpy array (range [-1, 1])."""
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    
    return audio


def _save_wav(path: Path, audio: np.ndarray, sr: int = 24000):
    """Save numpy array as 16-bit WAV."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    config: PrepConfig,
    skip_transcription: bool = False,
):
    """Run the complete data preparation pipeline."""
    
    logger.info("═" * 60)
    logger.info("  CALLEX TTS — Data Preparation Pipeline")
    logger.info("═" * 60)
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("")
    
    # Create output directories
    converted_dir = output_dir / "01_converted"
    segments_dir = output_dir / "02_agent_segments"
    denoised_dir = output_dir / "03_denoised"
    filtered_dir = output_dir / "04_filtered"
    
    for d in [converted_dir, segments_dir, denoised_dir, filtered_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Discover audio files
    audio_files = discover_audio_files(input_dir)
    if not audio_files:
        logger.error("No audio files found in %s", input_dir)
        return
    
    # Statistics
    stats = {
        "total_files": len(audio_files),
        "converted": 0,
        "total_segments": 0,
        "kept_segments": 0,
        "rejected_segments": 0,
        "total_duration_sec": 0,
        "kept_duration_sec": 0,
        "reject_reasons": {},
    }
    
    segment_counter = 0
    metadata_rows: list[tuple[str, str, str]] = []
    
    for file_idx, audio_file in enumerate(audio_files, 1):
        logger.info("━" * 50)
        logger.info("[%d/%d] Processing: %s", file_idx, len(audio_files), audio_file.name)
        
        # Step 1: Convert to WAV
        wav_path = converted_dir / f"{audio_file.stem}.wav"
        if not convert_to_wav(audio_file, wav_path, config.target_sr):
            logger.warning("  ⏭️ Skipping (conversion failed)")
            continue
        stats["converted"] += 1
        
        # Step 2: Diarize
        segments = diarize_audio(wav_path, config)
        if not segments:
            logger.warning("  ⏭️ No speech segments found")
            continue
        
        # Load full audio
        full_audio = _load_wav_numpy(wav_path)
        
        # Step 3-5: Process each segment
        for seg in segments:
            # Only keep target speaker (agent)
            if seg["speaker"] != config.speaker_to_keep:
                continue
            
            stats["total_segments"] += 1
            
            # Extract segment
            audio_seg = extract_segment(full_audio, seg["start"], seg["end"], config.target_sr)
            duration = len(audio_seg) / config.target_sr
            stats["total_duration_sec"] += duration
            
            # Quality check
            passed, reason = check_quality(audio_seg, config.target_sr, config)
            if not passed:
                stats["rejected_segments"] += 1
                stats["reject_reasons"][reason.split(" ")[0]] = stats["reject_reasons"].get(reason.split(" ")[0], 0) + 1
                continue
            
            # Denoise
            audio_clean = denoise_audio(audio_seg, config.target_sr)
            
            # Normalize amplitude
            peak = np.max(np.abs(audio_clean))
            if peak > 0:
                audio_clean = audio_clean / peak * 0.95
            
            # Save
            segment_counter += 1
            file_id = f"{config.file_prefix}_{segment_counter:07d}"
            seg_path = filtered_dir / f"{file_id}.wav"
            _save_wav(seg_path, audio_clean, config.target_sr)
            
            stats["kept_segments"] += 1
            stats["kept_duration_sec"] += duration
            
            # Step 6: Transcription
            if not skip_transcription:
                transcript = transcribe_segment(seg_path, config)
                if transcript:
                    metadata_rows.append((file_id, transcript, "default"))
                    logger.info("  ✅ %s (%.1fs): %s", file_id, duration, transcript[:60])
                else:
                    logger.warning("  ⚠️ %s: transcription empty", file_id)
            else:
                metadata_rows.append((file_id, "", "default"))
    
    # Step 7: Write metadata.csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for file_id, text, speaker in metadata_rows:
            f.write(f"{file_id}|{text}|{speaker}\n")
    
    # Report
    logger.info("")
    logger.info("═" * 60)
    logger.info("  PIPELINE COMPLETE — Results")
    logger.info("═" * 60)
    logger.info("  Files processed:    %d / %d", stats["converted"], stats["total_files"])
    logger.info("  Total segments:     %d", stats["total_segments"])
    logger.info("  Kept segments:      %d (%.0f%%)", stats["kept_segments"],
                 stats["kept_segments"] / max(stats["total_segments"], 1) * 100)
    logger.info("  Rejected segments:  %d", stats["rejected_segments"])
    logger.info("  Total audio:        %.1f min", stats["total_duration_sec"] / 60)
    logger.info("  Kept audio:         %.1f min", stats["kept_duration_sec"] / 60)
    logger.info("  Metadata:           %s", metadata_path)
    logger.info("")
    if stats["reject_reasons"]:
        logger.info("  Rejection reasons:")
        for reason, count in sorted(stats["reject_reasons"].items(), key=lambda x: -x[1]):
            logger.info("    %-20s %d", reason, count)
    logger.info("═" * 60)
    
    # Save stats
    stats_path = output_dir / "pipeline_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Callex TTS Data Preparation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory containing raw audio files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--speaker", default="SPEAKER_00",
        help="Speaker label to keep (default: SPEAKER_00 = first/louder speaker)",
    )
    parser.add_argument(
        "--whisper-url", default=None,
        help="URL of remote Whisper server (e.g., http://gpu:8123/transcribe)\n"
             "If not set, will try local Whisper model",
    )
    parser.add_argument(
        "--skip-transcription", action="store_true",
        help="Skip ASR transcription (you'll need to add transcripts manually)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=15.0,
        help="Maximum segment duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--min-snr", type=float, default=15.0,
        help="Minimum SNR in dB (default: 15.0)",
    )
    
    args = parser.parse_args()
    
    config = PrepConfig(
        speaker_to_keep=args.speaker,
        whisper_url=args.whisper_url,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        min_snr_db=args.min_snr,
    )
    
    run_pipeline(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        config=config,
        skip_transcription=args.skip_transcription,
    )


if __name__ == "__main__":
    main()
