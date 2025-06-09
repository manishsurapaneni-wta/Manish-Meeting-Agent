import json
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
from crewai import Crew, Task
from agents.summarizer import summarizer
from agents.decision_extractor import decision_extractor
from agents.action_tracker import action_tracker
from agents.followup_checker import followup_checker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingAnalyzer:
    def __init__(self, output_dir: str = "output", llm_model: str = "gpt-4"):
        """Initialize the meeting analyzer.
        
        Args:
            output_dir: Directory to save analysis results
            llm_model: LLM model to use for analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model
        
        # Initialize agents
        self.summarizer = summarizer
        self.decision_extractor = decision_extractor
        self.action_tracker = action_tracker
        self.followup_checker = followup_checker
        
    def analyze_meeting(self, transcript: Dict) -> Dict:
        """Run full meeting analysis using all agents.
        
        Args:
            transcript: Formatted transcript with speaker turns
            
        Returns:
            Complete meeting analysis
        """
        logger.info("Starting meeting analysis...")
        
        # Generate summary
        logger.info("Generating meeting summary...")
        summary = self.summarizer.summarize(transcript)
        
        # Extract decisions
        logger.info("Extracting decisions...")
        decisions = self.decision_extractor.extract_decisions(transcript)
        
        # Track action items
        logger.info("Tracking action items...")
        action_items = self.action_tracker.track_actions(transcript)
        
        # Check follow-ups
        logger.info("Checking follow-up items...")
        followups = self.followup_checker.check_followups(transcript)
        
        # Compile results
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "decisions": decisions,
            "action_items": action_items,
            "follow_ups": followups
        }
        
        return analysis
        
    def save_analysis(self, analysis: Dict, filename: str = "meeting_summary.json") -> str:
        """Save analysis results to JSON file.
        
        Args:
            analysis: Complete meeting analysis
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)
        return str(output_path)

def run_agents(transcript_text):
    """Run all CrewAI agents on the transcript.
    
    Args:
        transcript_text: Formatted transcript text
        
    Returns:
        Dictionary containing all analysis results
    """
    logger.info("Initializing CrewAI tasks...")
    
    tasks = [
        Task(
            description="Summarize this meeting in 5-10 bullet points.",
            agent=summarizer,
            input=transcript_text
        ),
        Task(
            description="List the decisions made in this meeting with speaker names.",
            agent=decision_extractor,
            input=transcript_text
        ),
        Task(
            description="Identify and assign action items from the discussion.",
            agent=action_tracker,
            input=transcript_text
        ),
        Task(
            description="Mention unresolved issues or follow-ups for the next meeting.",
            agent=followup_checker,
            input=transcript_text
        ),
    ]

    logger.info("Creating CrewAI crew...")
    crew = Crew(tasks=tasks)
    
    logger.info("Running analysis...")
    result = crew.run()
    
    return result

def main():
    """Example usage of MeetingAnalyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze meeting transcript")
    parser.add_argument("transcript_file", help="Path to formatted transcript JSON file")
    parser.add_argument("--output", default="meeting_summary.json", help="Output filename")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Load transcript
    with open(args.transcript_file) as f:
        transcript = json.load(f)
    
    # Run analysis
    analyzer = MeetingAnalyzer(llm_model=args.model)
    analysis = analyzer.analyze_meeting(transcript)
    
    # Save results
    output_path = analyzer.save_analysis(analysis, args.output)
    logger.info(f"Analysis saved to: {output_path}")

if __name__ == "__main__":
    main() 