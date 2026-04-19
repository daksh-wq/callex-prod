"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Dataset Builder                                       ║
║                                                                      ║
║  Takes processed audio from prepare_data.py and builds the final     ║
║  training-ready dataset with precomputed mel spectrograms.           ║
║                                                                      ║
║  Usage:                                                              ║
║    python scripts/build_dataset.py \                                 ║
║        --input data/04_processed/ \                                  ║
║        --output data/05_training_ready/                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("callex.data.build")


def build_dataset(input_dir: Path, output_dir: Path, sr: int = 24000):
    """Build training-ready dataset from processed audio."""
    
    logger.info("═" * 60)
    logger.info("  CALLEX TTS — Dataset Builder")
    logger.info("═" * 60)
    
    # Find metadata
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error("metadata.csv not found in %s", input_dir)
        return
    
    filtered_dir = input_dir / "04_filtered"
    if not filtered_dir.exists():
        # Try flat directory
        filtered_dir = input_dir
    
    # Output dirs
    wav_dir = output_dir / "wavs"
    mel_dir = output_dir / "mel_cache"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize mel extractor
    sys_path_hack()
    from callex_tts.audio.features import MelSpectrogramExtractor, AudioConfig
    
    mel_config = AudioConfig(sample_rate=sr, n_mels=80, hop_length=256)
    mel_extractor = MelSpectrogramExtractor(mel_config)
    
    # Process
    output_rows = []
    total_duration = 0.0
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
            
            file_id = parts[0].strip()
            text = parts[1].strip()
            speaker = parts[2].strip() if len(parts) > 2 else "default"
            
            # Skip empty transcriptions
            if not text:
                continue
            
            # Find source WAV
            src_wav = filtered_dir / f"{file_id}.wav"
            if not src_wav.exists():
                continue
            
            # Copy WAV to output
            dst_wav = wav_dir / f"{file_id}.wav"
            shutil.copy2(src_wav, dst_wav)
            
            # Compute and cache mel spectrogram
            try:
                import torchaudio
                waveform, wav_sr = torchaudio.load(str(dst_wav))
                
                if wav_sr != sr:
                    resampler = torchaudio.transforms.Resample(wav_sr, sr)
                    waveform = resampler(waveform)
                
                # Mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Normalize
                waveform = waveform / (waveform.abs().max() + 1e-6)
                
                # Compute mel
                mel = mel_extractor.mel_spectrogram(waveform).squeeze(0)
                
                # Save mel cache
                mel_path = mel_dir / f"{file_id}.pt"
                torch.save(mel, mel_path)
                
                duration = waveform.shape[-1] / sr
                total_duration += duration
                
                output_rows.append((file_id, text, speaker))
                
            except Exception as e:
                logger.warning("Failed to process %s: %s", file_id, e)
                continue
    
    # Write final metadata
    final_metadata = output_dir / "metadata.csv"
    with open(final_metadata, 'w', encoding='utf-8') as f:
        for file_id, text, speaker in output_rows:
            f.write(f"{file_id}|{text}|{speaker}\n")
    
    # Report
    logger.info("")
    logger.info("═" * 60)
    logger.info("  DATASET READY")
    logger.info("═" * 60)
    logger.info("  Utterances:    %d", len(output_rows))
    logger.info("  Total audio:   %.1f min (%.1f hours)", total_duration / 60, total_duration / 3600)
    logger.info("  WAVs:          %s", wav_dir)
    logger.info("  Mel cache:     %s", mel_dir)
    logger.info("  Metadata:      %s", final_metadata)
    logger.info("")
    logger.info("  Next step: Train the model!")
    logger.info("    make train")
    logger.info("    # or")
    logger.info("    python -m callex_tts.training.trainer --config configs/training/distributed.yaml")
    logger.info("═" * 60)


def sys_path_hack():
    """Add src/ to PYTHONPATH for imports."""
    import sys
    src = str(Path(__file__).parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main():
    parser = argparse.ArgumentParser(description="Build training-ready TTS dataset")
    parser.add_argument("--input", required=True, help="Processed data directory")
    parser.add_argument("--output", required=True, help="Training-ready output directory")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate")
    args = parser.parse_args()
    
    build_dataset(Path(args.input), Path(args.output), args.sr)


if __name__ == "__main__":
    main()
