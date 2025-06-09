from crewai import Agent
from typing import Dict, List

decision_extractor = Agent(
    name="Decision Extractor",
    role="Decision Analysis Expert",
    goal="Identify and extract all decisions made during meetings",
    backstory="""You are an expert at analyzing meeting discussions and identifying 
    explicit and implicit decisions made by participants. You have a keen eye for 
    spotting when consensus is reached or when key choices are made.""",
    verbose=True
)

class DecisionExtractorAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        """Initialize the decision extractor agent.
        
        Args:
            llm_model: LLM model to use for decision extraction
        """
        self.agent = Agent(
            role="Decision Extractor",
            goal="Identify and extract all decisions made during meetings",
            backstory="""You are an expert at analyzing meeting discussions and identifying 
            explicit and implicit decisions made by participants. You have a keen eye for 
            spotting when consensus is reached or when key choices are made.""",
            verbose=True,
            llm_model=llm_model
        )
        
    def extract_decisions(self, transcript: Dict) -> List[Dict]:
        """Extract decisions from the meeting transcript.
        
        Args:
            transcript: Formatted transcript with speaker turns
            
        Returns:
            List of decisions with context
        """
        # Prepare context for the agent
        context = {
            "full_text": transcript["full_text"],
            "speakers": transcript["speakers"],
            "segments": transcript["segments"]
        }
        
        # Create task for the agent
        task = self.agent.create_task(
            description=f"""Analyze this meeting transcript and extract all decisions made.
            For each decision, identify:
            1. The specific decision made
            2. Who made the decision
            3. The context and reasoning behind it
            4. Any conditions or caveats attached
            
            Format each decision as a JSON object with these fields.
            
            Transcript: {context}""",
            expected_output="A list of decision objects in JSON format"
        )
        
        # Execute task and get decisions
        decisions = self.agent.execute_task(task)
        return decisions

def main():
    """Example usage of DecisionExtractorAgent."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract decisions from meeting")
    parser.add_argument("transcript_file", help="Path to formatted transcript JSON file")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Load transcript
    with open(args.transcript_file) as f:
        transcript = json.load(f)
    
    # Extract decisions
    extractor = DecisionExtractorAgent(llm_model=args.model)
    decisions = extractor.extract_decisions(transcript)
    
    print("\nDecisions Made:")
    for decision in decisions:
        print(f"\nDecision: {decision['decision']}")
        print(f"Made by: {decision['made_by']}")
        print(f"Context: {decision['context']}")
        if decision.get('conditions'):
            print(f"Conditions: {decision['conditions']}")

if __name__ == "__main__":
    main() 