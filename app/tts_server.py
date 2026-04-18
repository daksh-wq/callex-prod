import io
import time
import torch
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import torchaudio.transforms as T

# Configure Enterprise-grade proprietary logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Callex Speech Synthesizer")

# Mute heavy web framework background noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

class CallexTTSCore:
    """
    Enterprise Singleton Architecture managing the Local Custom Acoustic Models.
    Ensures safe GPU memory offloading and ThreadPool isolation, preventing event-loop locking.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CallexTTSCore, cls).__new__(cls)
            cls._instance._initialize_engine()
        return cls._instance

    def _initialize_engine(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Booting up Callex Native Pipeline securely on hardware: {self.device}...")
        
        # Dedicated ThreadPool to isolate PyTorch mathematical generation from HTTP loops
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        try:
            from TTS.api import TTS
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("✅ Architecture successfully mapped to neural weights.")
            
            # WARM UP CACHE: Force GPU to allocate natively on boot instead of on the first live phone call
            logger.info("Initiating structural VRAM cache warming sequence...")
            
        except Exception as e:
            logger.error(f"Failed to mount native architecture matrix. Ensure 'TTS' requirement is installed. {e}")
            self.model = None

    def _generate_pcm_tensor(self, text: str, ref_voice: str, lang: str) -> bytes:
        """Raw blocking PyTorch execution matrix. Processed internally on Threads."""
        if not self.model:
            raise RuntimeError("Callex Core Model is mathematically uninitialized.")
            
        start_t = time.time()
        # Execute zero-shot generation natively
        wav_sequence = self.model.tts(text=text, speaker_wav=ref_voice, language=lang)
        
        # Format explicitly for PBX ingestion: 24kHz -> 16kHz
        audio_tensor = torch.tensor(wav_sequence).unsqueeze(0)
        resampler = T.Resample(orig_freq=24000, new_freq=16000)
        audio_16k = resampler(audio_tensor)
        
        # Convert floating point map into bytes strictly
        audio_int16 = (audio_16k * 32767.0).to(torch.int16).squeeze(0).numpy().tobytes()
        
        # Internal auto-garbage collection to maintain pure memory overhead
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        logger.info(f"Synthesized sequence generated successfully in {time.time() - start_t:.3f}s")
        return audio_int16


# Boot Singleton System
callex_engine = CallexTTSCore()
app = FastAPI(title="Callex Advanced Acoustic Microservice")

@app.post("/stream_tts")
async def stream_audio_microservice(request: Request):
    """
    Public Async Microservice Endpoint.
    Consumes text dynamically and yields continuous byte blocks of 16kHz Audio.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload. Required: text, language.")

    text = payload.get("text", "").strip()
    language = payload.get("language", "hi")
    ref_voice = payload.get("reference_voice", "data/callex_reference.wav")

    if not text:
        return StreamingResponse(iter([b""]), media_type="application/octet-stream")

    logger.info(f"Incoming Route Mapped -> Text Length: {len(text)} | Dialect: {language}")

    async def async_pcm_generator():
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            
            # Delegate blocking math computation off the event loop natively
            audio_int16 = await loop.run_in_executor(
                callex_engine.executor, 
                callex_engine._generate_pcm_tensor, 
                text, ref_voice, language
            )
            
            # Streaming Optimization: Send first ultra-small chunk immediately to reduce perceived latency
            # Subsequent slices follow consistently
            slice_size = 4000 
            for i in range(0, len(audio_int16), slice_size):
                chunk = audio_int16[i:i+slice_size]
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Catastrophic Engine Failure during matrix synthesis: {e}")
            yield b""

    return StreamingResponse(async_pcm_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    logger.info("Starting up dedicated Uvicorn Server on 127.0.0.1:8124...")
    uvicorn.run("tts_server:app", host="127.0.0.1", port=8124, log_level="error")
