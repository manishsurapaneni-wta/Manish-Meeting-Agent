import os
import argparse
import logging
from pathlib import Path
from typing import Optional

from scripts.whisper_transcribe import WhisperTranscriber
from scripts.format_transcript import TranscriptFormatter
from scripts.run_crewai_agents import MeetingAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeetingCopilot:
    def __init__(
        self,
        audio_dir: str = "audio",
        output_dir: str = "output",
        whisper_model: str = "large-v2",
        llm_model: str = "gpt-4",
        device: Optional[str] = None
    ):
        """Initialize the meeting copilot.
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory for output files
            whisper_model: Whisper model to use
            llm_model: LLM model to use for analysis
            device: Device to run Whisper on (cuda/cpu)
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.whisper_model = whisper_model
        self.llm_model = llm_model
        self.device = device
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.transcriber = WhisperTranscriber(
            model_name=whisper_model,
            device=device
        )
        self.formatter = TranscriptFormatter(output_dir=output_dir)
        self.analyzer = MeetingAnalyzer(
            output_dir=output_dir,
            llm_model=llm_model
        )
        
    def process_meeting(self, audio_file: str) -> str:
        """Process a meeting audio file through the full pipeline.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Path to final analysis file
        """
        audio_path = self.audio_dir / audio_file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Processing meeting: {audio_file}")
        
        # Step 1: Transcribe audio
        logger.info("Transcribing audio...")
        transcription = self.transcriber.transcribe(str(audio_path))
        
        # Step 2: Format transcript
        logger.info("Formatting transcript...")
        formatted_transcript = self.formatter.format_transcript(transcription)
        transcript_path = self.formatter.save_transcript(
            formatted_transcript,
            f"{audio_path.stem}_transcript.json"
        )
        
        # Step 3: Analyze meeting
        logger.info("Analyzing meeting...")
        analysis = self.analyzer.analyze_meeting(formatted_transcript)
        analysis_path = self.analyzer.save_analysis(
            analysis,
            f"{audio_path.stem}_analysis.json"
        )
        
        logger.info(f"Meeting processing complete. Analysis saved to: {analysis_path}")
        return analysis_path

def main():
    """Command-line interface for the meeting copilot."""
    parser = argparse.ArgumentParser(
        description="Process meeting audio and generate analysis"
    )
    parser.add_argument(
        "audio_file",
        help="Audio file to process (will be copied to audio directory)"
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v2",
        help="Whisper model to use"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="LLM model to use for analysis"
    )
    parser.add_argument(
        "--device",
        help="Device to run Whisper on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize copilot
    copilot = MeetingCopilot(
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        device=args.device
    )
    
    # Process meeting
    try:
        analysis_path = copilot.process_meeting(args.audio_file)
        print(f"\nMeeting analysis complete! Results saved to: {analysis_path}")
    except Exception as e:
        logger.error(f"Error processing meeting: {e}")
        raise

if __name__ == "__main__":
    main() 