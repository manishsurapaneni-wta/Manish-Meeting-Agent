from crewai import Agent
from typing import Dict, List

followup_checker = Agent(
    name="Follow-up Analyzer",
    role="Follow-up Analysis Expert",
    goal="Identify topics that need follow-up discussion or clarification",
    backstory="""You are an expert at analyzing meeting discussions and identifying 
    topics that need further discussion, clarification, or follow-up. You excel at 
    recognizing unresolved issues and areas that require additional attention.""",
    verbose=True
)

class FollowupCheckerAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        """Initialize the follow-up checker agent.
        
        Args:
            llm_model: LLM model to use for follow-up analysis
        """
        self.agent = Agent(
            role="Follow-up Analyzer",
            goal="Identify topics that need follow-up discussion or clarification",
            backstory="""You are an expert at analyzing meeting discussions and identifying 
            topics that need further discussion, clarification, or follow-up. You excel at 
            recognizing unresolved issues and areas that require additional attention.""",
            verbose=True,
            llm_model=llm_model
        )
        
    def check_followups(self, transcript: Dict) -> List[Dict]:
        """Extract follow-up items from the meeting transcript.
        
        Args:
            transcript: Formatted transcript with speaker turns
            
        Returns:
            List of follow-up items with details
        """
        # Prepare context for the agent
        context = {
            "full_text": transcript["full_text"],
            "speakers": transcript["speakers"],
            "segments": transcript["segments"]
        }
        
        # Create task for the agent
        task = self.agent.create_task(
            description=f"""Analyze this meeting transcript and identify topics that need follow-up.
            For each follow-up item, identify:
            1. The topic or issue that needs follow-up
            2. Why it needs follow-up (e.g., unresolved, needs clarification)
            3. Who should be involved in the follow-up
            4. Suggested timing or urgency
            5. Any specific questions or points to address
            
            Format each follow-up item as a JSON object with these fields.
            
            Transcript: {context}""",
            expected_output="A list of follow-up item objects in JSON format"
        )
        
        # Execute task and get follow-up items
        followups = self.agent.execute_task(task)
        return followups

def main():
    """Example usage of FollowupCheckerAgent."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Check follow-up items from meeting")
    parser.add_argument("transcript_file", help="Path to formatted transcript JSON file")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Load transcript
    with open(args.transcript_file) as f:
        transcript = json.load(f)
    
    # Check follow-ups
    checker = FollowupCheckerAgent(llm_model=args.model)
    followups = checker.check_followups(transcript)
    
    print("\nFollow-up Items:")
    for item in followups:
        print(f"\nTopic: {item['topic']}")
        print(f"Reason: {item['reason']}")
        print(f"Participants: {item['participants']}")
        if item.get('urgency'):
            print(f"Urgency: {item['urgency']}")
        if item.get('points_to_address'):
            print(f"Points to Address: {item['points_to_address']}")

if __name__ == "__main__":
    main() 