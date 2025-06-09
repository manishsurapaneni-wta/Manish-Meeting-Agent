import os
import torch
import whisperx
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_name: str = "large-v2", device: Optional[str] = None):
        """Initialize the WhisperX transcriber.
        
        Args:
            model_name: Whisper model to use
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load WhisperX model
        self.model = whisperx.load_model(
            model_name,
            self.device,
            compute_type="float16" if self.device == "cuda" else "float32"
        )
        
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio file with speaker diarization.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing transcription and speaker information
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Transcribing {audio_path}")
        
        # Transcribe audio
        result = self.model.transcribe(
            audio_path,
            batch_size=16,
            language="en"
        )
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning("No Hugging Face token found. Diarization may not work.")
            logger.warning("Set HUGGINGFACE_TOKEN in .env file or environment variables.")
        
        # Load diarization model
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=self.device
        )
        
        # Get diarization results
        diarize_segments = diarize_model(
            audio_path,
            min_speakers=1,
            max_speakers=10
        )
        
        # Assign speaker labels
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return {
            "segments": result["segments"],
            "speakers": result["speakers"],
            "text": result["text"]
        }

def main():
    """Example usage of WhisperTranscriber."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="large-v2", help="Whisper model to use")
    parser.add_argument("--device", help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()
    
    transcriber = WhisperTranscriber(model_name=args.model, device=args.device)
    result = transcriber.transcribe(args.audio_path)
    
    # Print transcription with speaker labels
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        print(f"[{speaker}] {segment['text']}")

if __name__ == "__main__":
    main() 