import chromadb
from chromadb.utils import embedding_functions
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingMemory:
    def __init__(self):
        """Initialize the meeting memory with ChromaDB."""
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        self.collection = self.client.get_or_create_collection(
            name="meeting_memories",
            embedding_function=self.embedding_function
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def add_meeting(self, summary_json, meeting_id=None):
        """Add a meeting summary to memory."""
        if meeting_id is None:
            meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process each section of the summary
        for section, content in summary_json.items():
            if isinstance(content, list):
                for item in content:
                    text = str(item)
                    # Extract speaker/owner information
                    speaker = item.get("speaker") or item.get("owner") or "Unknown"
                    self.collection.add(
                        documents=[text],
                        metadatas=[{
                            "meeting_id": meeting_id,
                            "section": section,
                            "speaker": speaker,
                            "timestamp": datetime.now().isoformat()
                        }],
                        ids=[f"{meeting_id}_{section}_{hash(text) % 100000}"]
                    )
            else:
                self.collection.add(
                    documents=[str(content)],
                    metadatas=[{
                        "meeting_id": meeting_id,
                        "section": section,
                        "speaker": "general",
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[f"{meeting_id}_{section}_{hash(str(content)) % 100000}"]
                )
        
        logger.info(f"Added meeting {meeting_id} to memory")
        return meeting_id

    def search_meetings(self, query, n_results=5):
        """Search through meeting memories."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return formatted_results

    def get_meeting_history(self, meeting_id):
        """Get all segments from a specific meeting."""
        results = self.collection.get(
            where={"meeting_id": meeting_id}
        )
        
        formatted_results = []
        for i in range(len(results["documents"])):
            formatted_results.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return formatted_results

    def get_speaker_history(self, speaker_name):
        """Get all segments from a specific speaker."""
        results = self.collection.get(
            where={"speaker": speaker_name}
        )
        
        formatted_results = []
        for i in range(len(results["documents"])):
            formatted_results.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return formatted_results

    def summarize_all_meetings(self):
        """Generate a summary of all meetings."""
        results = self.collection.get()
        if not results["documents"]:
            return "No meetings found in memory."

        # Combine all meeting content
        all_text = "\n".join(results["documents"])
        
        # Generate summary using GPT-4
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a meeting analyst. Create a comprehensive summary of the following meeting content, highlighting key decisions, action items, and important discussions."},
                {"role": "user", "content": f"Summarize the following meeting content:\n{all_text}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content

    def get_speaker_summary(self, speaker_name):
        """Generate a summary of all contributions from a specific speaker."""
        speaker_history = self.get_speaker_history(speaker_name)
        if not speaker_history:
            return f"No contributions found for {speaker_name}."

        # Combine all speaker contributions
        all_text = "\n".join([item["text"] for item in speaker_history])
        
        # Generate summary using GPT-4
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a meeting analyst. Create a comprehensive summary of {speaker_name}'s contributions across all meetings, highlighting their key decisions, action items, and important discussions."},
                {"role": "user", "content": f"Summarize the following contributions:\n{all_text}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content

def main():
    """Example usage of the MeetingMemory class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Memory Management")
    parser.add_argument("--action", choices=["add", "search", "history", "summary", "speaker"], required=True)
    parser.add_argument("--query", help="Search query or meeting ID")
    parser.add_argument("--file", help="JSON file containing meeting summary")
    parser.add_argument("--speaker", help="Speaker name for speaker-specific operations")
    
    args = parser.parse_args()
    memory = MeetingMemory()
    
    if args.action == "add" and args.file:
        with open(args.file, "r") as f:
            summary = json.load(f)
        meeting_id = memory.add_meeting(summary)
        print(f"Added meeting with ID: {meeting_id}")
    
    elif args.action == "search" and args.query:
        results = memory.search_meetings(args.query)
        print("\nSearch Results:")
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Meeting: {result['metadata']['meeting_id']}")
            print(f"Section: {result['metadata']['section']}")
            print(f"Speaker: {result['metadata']['speaker']}")
    
    elif args.action == "history" and args.query:
        results = memory.get_meeting_history(args.query)
        print(f"\nMeeting History for {args.query}:")
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Section: {result['metadata']['section']}")
            print(f"Speaker: {result['metadata']['speaker']}")
    
    elif args.action == "summary":
        summary = memory.summarize_all_meetings()
        print("\nAll Meetings Summary:")
        print(summary)
    
    elif args.action == "speaker" and args.speaker:
        summary = memory.get_speaker_summary(args.speaker)
        print(f"\nSpeaker Summary for {args.speaker}:")
        print(summary)

if __name__ == "__main__":
    main() 