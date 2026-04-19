# 📁 Callex TTS Training Data

## Folder Structure

```
data/
├── 01_raw_calls/           ← YOU UPLOAD HERE (raw call recordings)
│   ├── batch_001/          ← Group by date/batch
│   │   ├── call_001.wav
│   │   ├── call_002.mp3
│   │   └── ...
│   └── batch_002/
│
├── 02_studio_recordings/   ← YOU UPLOAD HERE (clean studio recordings)
│   ├── speaker_priya/
│   │   ├── priya_0001.wav
│   │   ├── priya_0002.wav
│   │   └── ...
│   └── speaker_rahul/
│
├── 03_scripts/             ← Text scripts for studio recording
│   ├── banking_scripts.txt
│   ├── insurance_scripts.txt
│   └── general_hindi.txt
│
├── 04_processed/           ← AUTO-GENERATED (pipeline output)
│   ├── agent_segments/     ← Extracted agent voice clips
│   ├── denoised/           ← Cleaned audio
│   ├── transcripts/        ← ASR transcriptions
│   └── filtered/           ← Quality-filtered final data
│
├── 05_training_ready/      ← AUTO-GENERATED (final training format)
│   ├── wavs/               ← Clean WAV files (24kHz, mono, 16-bit)
│   │   ├── CX_0000001.wav
│   │   ├── CX_0000002.wav
│   │   └── ...
│   ├── metadata.csv        ← file_id|transcription|speaker_id
│   └── mel_cache/          ← Precomputed mel spectrograms
│
├── scripts.txt             ← Recording scripts for your voice artist
├── recording_guide.md      ← How to record studio audio
└── README.md               ← This file
```

## Quick Start

1. Upload raw call recordings to `01_raw_calls/`
2. Run: `python scripts/prepare_data.py --input data/01_raw_calls/ --output data/04_processed/`
3. Review quality in `04_processed/filtered/`
4. Run: `python scripts/build_dataset.py --input data/04_processed/filtered/ --output data/05_training_ready/`
5. Train: `make train`
