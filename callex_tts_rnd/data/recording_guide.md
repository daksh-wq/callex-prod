# 🎙️ Callex TTS — Voice Recording Guide

## For Studio Recordings (Best Quality)

### Equipment Needed
- **Microphone**: Any decent USB condenser mic (Blue Yeti, AT2020, or even a good phone in a quiet room)
- **Room**: Quiet room, no AC hum, no traffic noise, no echo
- **Software**: Audacity (free) or any recorder that outputs WAV

### Recording Settings
| Setting | Value |
|---------|-------|
| Sample Rate | **24000 Hz** (24kHz) or 44100 Hz (we'll downsample) |
| Bit Depth | 16-bit |
| Channels | Mono |
| Format | WAV (preferred) or MP3 (we'll convert) |
| Distance from mic | 15-20 cm (6-8 inches) |

### Recording Instructions for Voice Artist

1. **Read naturally** — speak like you're talking to a customer on phone, NOT like reading a book
2. **Maintain consistent volume** — don't whisper some lines and shout others
3. **Pause 1-2 seconds** between each sentence
4. **If you make a mistake**, pause 3 seconds, then re-read that sentence
5. **Stay hydrated** — drink water every 15-20 minutes
6. **Record in sessions** of 30-45 minutes max (voice fatigue kills quality)
7. **Same mic position** every session — consistency is critical

### File Naming
```
speaker_name_0001.wav
speaker_name_0002.wav
...
```

### What to Read
Use the scripts in `data/03_scripts/`. If you don't have scripts yet, the `scripts.txt` 
file in this directory has 500+ ready-to-read Hindi sentences covering banking, insurance, 
greetings, and general conversation.

---

## For Call Center Recordings

### What We Accept
- Any audio format: WAV, MP3, OGG, FLAC, M4A
- Any sample rate (we'll resample)
- Stereo or mono (we'll extract/mix)
- Any duration (we'll segment)

### What Makes Good Training Data
| ✅ Good | ❌ Bad |
|---------|--------|
| Clear agent voice | Heavy cross-talk |
| Minimal background noise | Loud office environment |
| Hindi/Hinglish speech | Music on hold |
| Normal speaking pace | Shouting or whispering |
| 8kHz+ quality | Extremely compressed audio |

### Upload Recommendations
- **Group by batch/date** in subfolders
- **Label the agent** if you know who it is (helps us train single-speaker models)
- **Minimum**: 50 calls (~5-10 hours of audio)
- **Ideal**: 200+ calls (~20-50 hours of audio)
- **Best**: 500+ calls + 5 hours studio recordings from the target voice

---

## Data Privacy

⚠️ **IMPORTANT**: Call recordings contain customer PII.

Before uploading:
1. Ensure you have legal consent to use recordings for AI training
2. We process ONLY the agent's voice — customer audio is discarded
3. No transcriptions of customer speech are stored
4. All intermediate processing files are deleted after dataset creation
