"""
SSTModel2StreamingSTT — Production WebSocket Streaming STT Client
================================================================

Maintains a persistent WebSocket connection to SSTModel2 AI's real-time
speech-to-text API.

Architecture:
  - Audio is sent as base64-encoded PCM16 JSON messages
  - Server returns transcript events, speech_start, and speech_end signals
  - Callbacks fire on the asyncio event loop for zero-blocking integration

Usage:
    stt = SSTModel2StreamingSTT(
        api_key="...",
        on_transcript=my_handler,      # async def (text: str)
        on_speech_started=my_handler,  # async def ()
        on_speech_ended=my_handler,    # async def ()
    )
    await stt.connect()
    stt.send_audio(pcm16_bytes)
    await stt.disconnect()
"""

import asyncio
import base64
import json

from base64 import b64decode as _b64

def __safe_log(msg) -> str:
    import builtins
    if msg is None: return "None"
    s = builtins.str(msg)
    # Sanitization patterns (obfuscated)
    _p = {
        _b64("c2FydmFt").decode(): "cx-stt", _b64("U2FydmFt").decode(): "CX-STT", _b64("U0FSVkFN").decode(): "CX-STT",
        _b64("c2FhcmFz").decode(): "cx-asr-core", _b64("U2FhcmFz").decode(): "CX-ASR-Core",
        _b64("Z2VtaW5p").decode(): "cx-model", _b64("R2VtaW5p").decode(): "CX-Model", _b64("R0VNSU5J").decode(): "CX-MODEL",
        _b64("ZWxldmVubGFicw==").decode(): "cx-voice", _b64("RWxldmVuTGFicw==").decode(): "CX-Voice", _b64("RUxFVkVOTEFCUw==").decode(): "CX-VOICE",
        _b64("Z2VuZXJhdGl2ZWxhbmd1YWdlLmdvb2dsZWFwaXMuY29t").decode(): "llm-api.callex.ai",
        _b64("YXBpLnNhcnZhbS5haQ==").decode(): "stt-api.callex.ai",
        _b64("YXBpLmVsZXZlbmxhYnMuaW8=").decode(): "voice-api.callex.ai",
        _b64("R29vZ2xlIEdlbkFJ").decode(): "CX-AI-Engine"
    }
    for k, v in _p.items(): s = s.replace(k, v)
    return s

import struct
import time
from typing import Callable, Optional, Awaitable, Any

import websockets
import websockets.exceptions


class SSTModel2StreamingSTT:
    """
    Production WebSocket client for SSTModel2 AI Streaming STT.
    
    Thread-safe: all public methods are safe to call from the asyncio event loop.
    Audio sending is non-blocking (fire-and-forget into the WS send buffer).
    """

    import base64
    WS_URL = base64.b64decode(b'd3NzOi8vYXBpLnNhcnZhbS5haS9zcGVlY2gtdG8tdGV4dC93cw==').decode('utf-8')
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 1.0  # seconds between reconnect attempts

    def __init__(
        self,
        api_key: str = None,
        on_transcript: Callable[[str], Awaitable[None]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_ended: Optional[Callable[[], Awaitable[None]]] = None,
        model: str = "genartml-callex",
        language: str = "hi-IN",
        mode: str = "transcribe",
        sample_rate: int = 16000,
        vad_signals: bool = True,
        high_vad_sensitivity: bool = True,
        key_manager: Any = None,
    ):
        self._api_key = api_key
        self._key_manager = key_manager
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._model = model
        self._language = language
        self._mode = mode
        self._sample_rate = sample_rate
        self._vad_signals = vad_signals
        self._high_vad_sensitivity = high_vad_sensitivity

        self._ws = None
        self._receive_task: Optional[asyncio.Task] = None
        self._is_connected = False
        self._connect_time: Optional[float] = None
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._ws is not None

    async def connect(self):
        """Establish the WebSocket connection to SSTModel2 streaming API with Key Rotation."""
        import base64
        actual_model = base64.b64decode(b'c2FhcmFzOnYz').decode('utf-8') if self._model == "genartml-callex" else self._model
        
        params = [
            f"model={actual_model}",
            f"language-code={self._language}",
            f"mode={self._mode}",
            f"sample_rate={self._sample_rate}",
            "input_audio_codec=wav",
        ]
        if self._vad_signals:
            params.append("vad_signals=true")
        if self._high_vad_sensitivity:
            params.append("high_vad_sensitivity=true")

        url = f"{self.WS_URL}?{'&'.join(params)}"

        if self._key_manager:
            primary_key = self._key_manager.get_key()
            keys_to_try = [primary_key] + self._key_manager.get_all_keys_for_retry(exclude_key=primary_key)
        else:
            keys_to_try = [self._api_key]

        last_error = None
        
        for attempt, key in enumerate(keys_to_try):
            if not key:
                continue
            headers = {
                "Api-Subscription-Key": key,
            }
            try:
                self._ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**20,  # 1MB max message
                )
                self._is_connected = True
                self._connect_time = time.time()
                self._reconnect_count = 0
                self._api_key = key  # lock in successful key

                # Start background receiver
                self._receive_task = asyncio.create_task(self._receive_loop())
                print(f"[SST_MODEL_2 WS] ✅ Connected to streaming STT ({self._model}, {self._language}) using key #{attempt + 1}")
                return

            except websockets.exceptions.InvalidStatus as e:
                last_error = e
                status_code = e.response.status_code
                print(f"[SST_MODEL_2 WS] ⚠️ Key #{attempt + 1} HTTP {status_code} ({key[:8]}...)")
                if self._key_manager:
                    self._key_manager.report_failure(key, status_code)
                    if attempt < len(keys_to_try) - 1:
                        print(f"[SST_MODEL_2 WS] 🔄 Retrying connection with next available API key...")
            except Exception as e:
                last_error = e
                print(f"[SST_MODEL_2 WS] ⚠️ Connection failed on key #{attempt+1}: {__safe_log(e)}")
                if attempt < len(keys_to_try) - 1:
                    print(f"[SST_MODEL_2 WS] 🔄 Retrying connection with next available API key...")

        self._is_connected = False
        print(f"[SST_MODEL_2 WS] ❌ ALL connection attempts to SSTModel2 streaming dropped/failed! Fallback ASR required. Last Error: {__safe_log(last_error)}")
        raise last_error

    async def _receive_loop(self):
        """Background task: reads messages from SSTModel2 WS and dispatches callbacks."""
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    if msg_type == "speech_start":
                        if self._on_speech_started:
                            await self._on_speech_started()

                    elif msg_type == "speech_end":
                        if self._on_speech_ended:
                            await self._on_speech_ended()

                    elif msg_type == "data":
                        # Transcript result
                        data = msg.get("data", {})
                        transcript = data.get("transcript", "").strip()
                        if transcript:
                            metrics = data.get("metrics", {})
                            latency = metrics.get("processing_latency", 0)
                            audio_dur = metrics.get("audio_duration", 0)
                            print(f"[SST_MODEL_2 WS] 📝 Transcript: '{transcript[:80]}' (latency={latency:.2f}s, audio={audio_dur:.2f}s)")
                            await self._on_transcript(transcript)

                    elif msg_type == "error":
                        error_data = msg.get("data", {})
                        error_msg = error_data.get("message", __safe_log(msg))
                        print(f"[SST_MODEL_2 WS] ⚠️ Server error: {__safe_log(error_msg)}")

                    # Silently ignore unknown message types (heartbeats, etc.)

                except json.JSONDecodeError:
                    print(f"[SST_MODEL_2 WS] ⚠️ Non-JSON message: {__safe_log(raw_msg)[:100]}")
                except Exception as e:
                    print(f"[SST_MODEL_2 WS] ⚠️ Message handler error: {__safe_log(e)}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[SST_MODEL_2 WS] 🔌 Connection closed: {__safe_log(e)}")
            self._is_connected = False
            # Attempt reconnect
            await self._try_reconnect()

        except asyncio.CancelledError:
            # Clean shutdown — don't reconnect
            raise

        except Exception as e:
            print(f"[SST_MODEL_2 WS] ❌ Receive loop error: {__safe_log(e)}")
            self._is_connected = False
            await self._try_reconnect()

    async def _try_reconnect(self):
        """Attempt to reconnect after unexpected disconnect."""
        if self._reconnect_count >= self.MAX_RECONNECT_ATTEMPTS:
            print(f"[SST_MODEL_2 WS] ❌ Max reconnect attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
            return

        self._reconnect_count += 1
        print(f"[SST_MODEL_2 WS] 🔄 Reconnect attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS}...")
        await asyncio.sleep(self.RECONNECT_DELAY)

        try:
            await self.connect()
            print(f"[SST_MODEL_2 WS] ✅ Reconnected successfully")
        except Exception as e:
            print(f"[SST_MODEL_2 WS] ❌ Reconnect failed: {__safe_log(e)}")

    def send_audio(self, pcm16_bytes: bytes):
        """
        Send PCM16 audio to SSTModel2 streaming STT as binary WAV frames.
        
        SSTModel2's current WebSocket API expects raw binary WebSocket frames
        containing WAV-formatted audio data. We wrap raw PCM16 in a minimal
        WAV header and send it as a binary frame for maximum efficiency.
        
        Non-blocking: schedules the send on the event loop.
        Safe to call from the audio processing hot path.
        """
        if not self._is_connected or self._ws is None:
            return
        if not pcm16_bytes or len(pcm16_bytes) == 0:
            return

        try:
            # Build a minimal WAV header for this chunk
            num_channels = 1
            sample_width = 2  # 16-bit = 2 bytes
            data_size = len(pcm16_bytes)
            byte_rate = self._sample_rate * num_channels * sample_width
            block_align = num_channels * sample_width

            wav_header = struct.pack(
                '<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                36 + data_size,
                b'WAVE',
                b'fmt ',
                16,                     # Subchunk1Size (PCM)
                1,                      # AudioFormat (1 = PCM)
                num_channels,
                self._sample_rate,
                byte_rate,
                block_align,
                16,                     # BitsPerSample
                b'data',
                data_size
            )
            wav_bytes = wav_header + pcm16_bytes

            # Send as binary WebSocket frame (SSTModel2 expects raw binary audio)
            asyncio.create_task(self._safe_send_binary(wav_bytes))
        except Exception as e:
            print(f"[SST_MODEL_2 WS] ⚠️ send_audio error: {__safe_log(e)}")

    def send_flush(self):
        """Send flush signal to finalize transcript (SSTModel2 API format)."""
        if not self._is_connected or self._ws is None:
            return
        try:
            message = json.dumps({"type": "flush"})
            asyncio.create_task(self._safe_send(message))
        except Exception as e:
            print(f"[SST_MODEL_2 WS] ⚠️ send_flush error: {__safe_log(e)}")

    async def _safe_send(self, message: str):
        """Send a text message with error handling (no crash on closed socket)."""
        try:
            if self._ws and self._is_connected:
                await self._ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except Exception as e:
            print(f"[SST_MODEL_2 WS] ⚠️ Send failed: {__safe_log(e)}")
            self._is_connected = False

    async def _safe_send_binary(self, data: bytes):
        """Send a binary frame with error handling (no crash on closed socket)."""
        try:
            if self._ws and self._is_connected:
                await self._ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            self._is_connected = False
        except Exception as e:
            print(f"[SST_MODEL_2 WS] ⚠️ Binary send failed: {__safe_log(e)}")
            self._is_connected = False

    async def disconnect(self):
        """Cleanly disconnect from SSTModel2 streaming API."""
        self._is_connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        duration = time.time() - self._connect_time if self._connect_time else 0
        print(f"[SST_MODEL_2 WS] 🔌 Disconnected (session duration: {duration:.0f}s)")
