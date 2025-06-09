from typing import Dict, List
import json
from datetime import timedelta
from pathlib import Path

class TranscriptFormatter:
    def __init__(self, output_dir: str = "output"):
        """Initialize the transcript formatter.
        
        Args:
            output_dir: Directory to save formatted transcripts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        return str(timedelta(seconds=int(seconds)))
        
    def format_transcript(self, transcription: Dict) -> Dict:
        """Format transcription into a structured format.
        
        Args:
            transcription: Raw transcription from WhisperX
            
        Returns:
            Formatted transcript with speaker turns and timestamps
        """
        formatted_segments = []
        
        for segment in transcription["segments"]:
            formatted_segment = {
                "speaker": segment.get("speaker", "UNKNOWN"),
                "start_time": self.format_time(segment["start"]),
                "end_time": self.format_time(segment["end"]),
                "text": segment["text"].strip()
            }
            formatted_segments.append(formatted_segment)
            
        return {
            "segments": formatted_segments,
            "full_text": transcription["text"],
            "speakers": transcription.get("speakers", [])
        }
        
    def save_transcript(self, formatted_transcript: Dict, filename: str) -> str:
        """Save formatted transcript to JSON file.
        
        Args:
            formatted_transcript: Formatted transcript
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(formatted_transcript, f, indent=2)
        return str(output_path)

def format_transcript(segments):
    """Format transcript segments into a readable text format.
    
    Args:
        segments: List of transcript segments with speaker and timing info
        
    Returns:
        Formatted transcript text
    """
    formatted = ""
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"]
        formatted += f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n"
    return formatted

def main():
    """Example usage of TranscriptFormatter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Format transcription output")
    parser.add_argument("transcript_file", help="Path to raw transcript JSON file")
    parser.add_argument("--output", default="formatted_transcript.json", help="Output filename")
    
    args = parser.parse_args()
    
    # Load raw transcript
    with open(args.transcript_file) as f:
        transcription = json.load(f)
    
    # Format and save
    formatter = TranscriptFormatter()
    formatted = formatter.format_transcript(transcription)
    output_path = formatter.save_transcript(formatted, args.output)
    
    print(f"Formatted transcript saved to: {output_path}")

if __name__ == "__main__":
    main() 