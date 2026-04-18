"""
Configuration Manager for Voice AI Bot
Handles loading, saving, and validating bot configuration settings
"""
import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class VADSettings(BaseModel):
    """Voice Activity Detection configuration"""
    min_speech_duration: float = Field(0.1, ge=0.05, le=2.0, description="Minimum speech duration in seconds")
    silence_timeout: float = Field(1.5, ge=0.3, le=5.0, description="Silence timeout in seconds")
    interruption_threshold_db: float = Field(-35.0, ge=-50.0, le=-10.0, description="Audio threshold in dB")
    noise_gate_db: float = Field(-55.0, ge=-70.0, le=-30.0, description="Noise gate threshold in dB")
    spectral_flatness_threshold: float = Field(0.55, ge=0.1, le=1.0, description="Spectral flatness threshold")


class APICredentials(BaseModel):
    """Callex Voice Engine API credentials"""
    server_key: str = Field(..., min_length=10, description="LLM API key")
    secret_key: str = Field(..., min_length=10, description="Callex Voice Engine secret key")
    voice_id: str = Field(..., min_length=10, description="Callex Voice ID")


class VoiceSettings(BaseModel):
    """Voice synthesis parameters"""
    speed: float = Field(1.25, ge=0.5, le=2.0, description="Speaking speed multiplier")
    stability: float = Field(0.85, ge=0.0, le=1.0, description="Voice stability (0=expressive, 1=consistent). High = locked pitch/tone")
    similarity_boost: float = Field(0.80, ge=0.0, le=1.0, description="How closely to match voice model")
    style: float = Field(0.0, ge=0.0, le=1.0, description="Speaking style intensity (0=stable, keep low for consistency)")
    use_speaker_boost: bool = Field(True, description="Enable speaker boost")


class BotConfig(BaseModel):
    """Complete bot configuration"""
    vad: VADSettings
    api_credentials: APICredentials
    voice: VoiceSettings


class ConfigManager:
    """Manages bot configuration file operations"""
    
    def __init__(self, config_path: str = "bot_config.json"):
        self.config_path = config_path
        self.config: Optional[BotConfig] = None
        
    def load_config(self) -> BotConfig:
        """Load configuration from file or environment"""
        if os.path.exists(self.config_path):
            print(f"[CONFIG] Loading from {self.config_path}")
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            self.config = BotConfig(**data)
            print(f"[CONFIG] Loaded from file")
        else:
            print(f"[CONFIG] No config file found, using defaults from code")
            self.config = self._load_from_environment()
        
        return self.config
    
    def save_config(self, config: BotConfig) -> None:
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        print(f"[CONFIG] Saved to {self.config_path}")
    
    def update_settings(self, updates: Dict[str, Any]) -> BotConfig:
        """Update specific settings and save to file"""
        if not self.config:
            self.config = self.load_config()
        
        current_dict = self.config.dict()
        
        # Deep merge updates
        for key, value in updates.items():
            if key in current_dict and isinstance(value, dict):
                current_dict[key].update(value)
            else:
                current_dict[key] = value
        
        # Validate and save
        self.config = BotConfig(**current_dict)
        self.save_config(self.config)
        
        return self.config
    
    def _load_from_environment(self) -> BotConfig:
        """Load configuration from environment/defaults"""
        # We cannot import test.py here because test.py imports ConfigManager,
        # which creates a circular dependency (RecursionError).
        # We will rely entirely on environment variables or defaults.
        
        # Return defaults from environment variables
        return BotConfig(
            vad=VADSettings(),
            api_credentials=APICredentials(
                server_key=os.getenv("CX_LLM_KEY") or os.getenv("GENARTML_SERVER_KEY", "set-your-token"),
                secret_key=os.getenv("GENARTML_SECRET_KEY", "set-your-callex-voice-key"),
                voice_id=os.getenv("GENARTML_VOICE_ID", "set-your-voice-id")
            ),
            voice=VoiceSettings()
        )
    
    def get_env_dict(self) -> Dict[str, Any]:
        """Get configuration as flat dictionary for environment variables"""
        if not self.config:
            self.config = self.load_config()
        
        return {
            'MIN_SPEECH_DURATION': self.config.vad.min_speech_duration,
            'SILENCE_TIMEOUT': self.config.vad.silence_timeout,
            'INTERRUPTION_THRESHOLD_DB': self.config.vad.interruption_threshold_db,
            'NOISE_GATE_DB': self.config.vad.noise_gate_db,
            'SPECTRAL_FLATNESS_THRESHOLD': self.config.vad.spectral_flatness_threshold,
            'GENARTML_SERVER_KEY': self.config.api_credentials.server_key,
            'GENARTML_SECRET_KEY': self.config.api_credentials.secret_key,
            'GENARTML_VOICE_ID': self.config.api_credentials.voice_id,
            'VOICE_SPEED': self.config.voice.speed,
            'VOICE_STABILITY': self.config.voice.stability,
            'VOICE_SIMILARITY_BOOST': self.config.voice.similarity_boost,
            'VOICE_STYLE': self.config.voice.style,
        }


# Global instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
