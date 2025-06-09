from crewai import Agent
from typing import Dict, List

action_tracker = Agent(
    name="Action Tracker",
    role="Action Item Specialist",
    goal="Identify and track all action items assigned during meetings",
    backstory="""You are an expert at identifying action items and tasks assigned 
    during meetings. You excel at recognizing both explicit assignments and implicit 
    responsibilities, and can determine clear owners and deadlines.""",
    verbose=True
)

class ActionTrackerAgent:
    def __init__(self, llm_model: str = "gpt-4"):
        """Initialize the action tracker agent.
        
        Args:
            llm_model: LLM model to use for action tracking
        """
        self.agent = Agent(
            role="Action Item Tracker",
            goal="Identify and track all action items assigned during meetings",
            backstory="""You are an expert at identifying action items and tasks assigned 
            during meetings. You excel at recognizing both explicit assignments and implicit 
            responsibilities, and can determine clear owners and deadlines.""",
            verbose=True,
            llm_model=llm_model
        )
        
    def track_actions(self, transcript: Dict) -> List[Dict]:
        """Extract action items from the meeting transcript.
        
        Args:
            transcript: Formatted transcript with speaker turns
            
        Returns:
            List of action items with details
        """
        # Prepare context for the agent
        context = {
            "full_text": transcript["full_text"],
            "speakers": transcript["speakers"],
            "segments": transcript["segments"]
        }
        
        # Create task for the agent
        task = self.agent.create_task(
            description=f"""Analyze this meeting transcript and extract all action items.
            For each action item, identify:
            1. The specific task or action required
            2. Who is responsible for it
            3. Any mentioned deadlines or timeframes
            4. Dependencies or prerequisites
            5. Priority level (if mentioned)
            
            Format each action item as a JSON object with these fields.
            
            Transcript: {context}""",
            expected_output="A list of action item objects in JSON format"
        )
        
        # Execute task and get action items
        action_items = self.agent.execute_task(task)
        return action_items

def main():
    """Example usage of ActionTrackerAgent."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Track action items from meeting")
    parser.add_argument("transcript_file", help="Path to formatted transcript JSON file")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    
    args = parser.parse_args()
    
    # Load transcript
    with open(args.transcript_file) as f:
        transcript = json.load(f)
    
    # Track action items
    tracker = ActionTrackerAgent(llm_model=args.model)
    action_items = tracker.track_actions(transcript)
    
    print("\nAction Items:")
    for item in action_items:
        print(f"\nTask: {item['task']}")
        print(f"Owner: {item['owner']}")
        if item.get('deadline'):
            print(f"Deadline: {item['deadline']}")
        if item.get('dependencies'):
            print(f"Dependencies: {item['dependencies']}")
        if item.get('priority'):
            print(f"Priority: {item['priority']}")

if __name__ == "__main__":
    main() 