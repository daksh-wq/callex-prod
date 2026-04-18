# Callex Local STT Architecture

This document outlines how the Callex Engine routes audio streams to the purely local Callex STT engine, bypassing external API endpoints or Node.js relays.

## 1. Why Removing the Node.js Route is Safe
In the legacy architecture, the Node.js server (`enterprise/backend/src/ws/sarvam.js`) acted strictly as a **Proxy** for the frontend dashboard simulator. Real phone calls *never* went through the Node.js server; they have always connected directly to the Python engine. 

To prevent connection errors when the web dashboard simulator attempts to connect to the legacy proxy, the Node.js server now returns a **graceful WebSocket close command** (`ws.close(1011, "Migrated")`). This safely rejects the legacy connection without crashing the frontend.

## 2. Where is the STT Engine Located?
The new STT engine operates natively within the Python AI system, completely decoupled from the Node.js proxy environment.
- **File Location:** `app/audio/callex_stt.py`
- **Underlying Technology:** `webrtcvad` handles sub-millisecond silence detection, routing speech chunks to a `Faster-Whisper` model running securely on local hardware via the `CallexSTT` integration class.

## 3. Active Telephony Routing Path
The sequence detailed below describes how voice audio is transcribed during a live call:

```mermaid
graph TD
    A[📞 Customer Phone] -->|Raw VoIP WSS| B[🐍 app/main.py WebSocket]
    
    subgraph Python Backend [Python Voice Engine]
        B --> C[Noise Filter & Silero VAD]
        C -->|Clean Audio Buffer| D(callex_stt.send_audio)
        
        subgraph Local STT Module ["📁 app/audio/callex_stt.py"]
            D --> E[chunking via webrtcvad]
            E -->|Speech Chunk| F[{Faster-Whisper Model Server}]
        end
        
        F -.->|Transcribed Text| G(on_callex_transcript)
        G --> H[LLM Chatbot]
    end
```

### The Code Execution Trace (Inside `app/main.py`)
At approximately Line 2820 inside `app/main.py`, the incoming WSS audio from FreeSWITCH reaches the Callex engine:

```python
# ── Feed VERIFIED audio to Callex STT ──
if callex_stt and callex_stt.is_connected and first_line_complete:
        callex_stt.send_audio(clean_int16.tobytes())
```

Once `send_audio()` fires, the byte allocation is instantaneously passed to the `app/audio/callex_stt.py` module. The local model transcribes the speech asynchronously without external networking, and triggers the predefined callback `_on_callex_transcript()` on Line 1982 to push the finalized transcript down to the LLM agent.
