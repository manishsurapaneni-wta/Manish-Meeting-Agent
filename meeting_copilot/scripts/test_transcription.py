import os
import json
import logging
from pathlib import Path
from whisper_transcribe import WhisperTranscriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transcription(audio_path: str, output_dir: str = "output"):
    """Test the transcription functionality.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save test results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize transcriber
    logger.info("Initializing WhisperX transcriber...")
    transcriber = WhisperTranscriber()
    
    # Transcribe audio
    logger.info(f"Transcribing {audio_path}...")
    result = transcriber.transcribe(audio_path)
    
    # Save raw result
    output_file = output_path / "test_transcription_raw.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Raw transcription saved to: {output_file}")
    
    # Print sample segments
    logger.info("\nSample transcription segments:")
    for segment in result["segments"][:5]:  # Print first 5 segments
        speaker = segment.get("speaker", "UNKNOWN")
        print(f"[{speaker}] {segment['text']}")
    
    # Print statistics
    total_segments = len(result["segments"])
    unique_speakers = len(set(seg.get("speaker", "UNKNOWN") for seg in result["segments"]))
    logger.info(f"\nTranscription Statistics:")
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Unique speakers: {unique_speakers}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WhisperX transcription")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        exit(1)
        
    test_transcription(args.audio_file, args.output) 