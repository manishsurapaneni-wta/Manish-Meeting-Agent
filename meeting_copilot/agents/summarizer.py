from crewai import Agent
from typing import Dict

summarizer = Agent(
    name="Summarizer",
    role="Meeting Summary Expert",
    goal="Condense the meeting into a concise summary",
    backstory="""You are an expert at analyzing meeting transcripts and creating 
    clear, structured summaries that capture the key points and main discussion topics.
    You have years of experience in business communication and meeting facilitation.""",
    verbose=True
)

class SummarizerAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        """Initialize the summarizer agent.
        
        Args:
            llm_model: LLM model to use for summarization
        """
        self.agent = Agent(
            role="Meeting Summarizer",
            goal="Create concise and comprehensive summaries of meeting transcripts",
            backstory="""You are an expert at analyzing meeting transcripts and creating 
            clear, structured summaries that capture the key points and main discussion topics.
            You have years of experience in business communication and meeting facilitation.""",
            verbose=True,
            llm_model=llm_model
        )
        
    def summarize(self, transcript: Dict) -> str:
        """Generate a summary of the meeting transcript.
        
        Args:
            transcript: Formatted transcript with speaker turns
            
        Returns:
            Meeting summary
        """
        # Prepare context for the agent
        context = {
            "full_text": transcript["full_text"],
            "speakers": transcript["speakers"],
            "segments": transcript["segments"]
        }
        
        # Create task for the agent
        task = self.agent.create_task(
            description=f"""Analyze this meeting transcript and create a comprehensive summary.
            Focus on:
            1. Main topics discussed
            2. Key points raised by each speaker
            3. Overall meeting flow and progression
            4. Important context and background information
            
            Transcript: {context}""",
            expected_output="A well-structured summary of the meeting in markdown format"
        )
        
        # Execute task and get summary
        summary = self.agent.execute_task(task)
        return summary

def main():
    """Example usage of SummarizerAgent."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate meeting summary")
    parser.add_argument("transcript_file", help="Path to formatted transcript JSON file")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Load transcript
    with open(args.transcript_file) as f:
        transcript = json.load(f)
    
    # Generate summary
    summarizer = SummarizerAgent(llm_model=args.model)
    summary = summarizer.summarize(transcript)
    
    print("\nMeeting Summary:")
    print(summary)

if __name__ == "__main__":
    main() 