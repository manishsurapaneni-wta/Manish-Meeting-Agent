from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import logging
from datetime import datetime
import json
from scripts.transcribe import WhisperTranscriber
from scripts.vector_memory import MeetingMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Meeting Copilot")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Initialize components
transcriber = WhisperTranscriber()
memory = MeetingMemory()

# Create necessary directories
os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Render the upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and processing."""
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"audio/{timestamp}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate meeting ID
        meeting_id = f"meeting_{timestamp}"
        
        # Transcribe audio
        logger.info(f"Transcribing {file_path}")
        transcript = transcriber.transcribe(file_path)
        
        # Save transcript
        transcript_path = f"output/{meeting_id}_transcript.txt"
        with open(transcript_path, "w") as f:
            f.write(transcript)
        
        # Format transcript
        formatted_transcript = transcriber.format_transcript(transcript)
        formatted_path = f"output/{meeting_id}_formatted.txt"
        with open(formatted_path, "w") as f:
            f.write(formatted_transcript)
        
        # Analyze meeting
        logger.info("Analyzing meeting content")
        analysis = transcriber.analyze_meeting(formatted_transcript)
        
        # Add timestamps to decisions and action items
        for segment in transcript.get("segments", []):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").lower()
            
            # Match decisions
            for decision in analysis.get("decisions", []):
                if decision["text"].lower() in text:
                    decision["start_time"] = start_time
                    decision["end_time"] = end_time
            
            # Match action items
            for item in analysis.get("action_items", []):
                if item["text"].lower() in text:
                    item["start_time"] = start_time
                    item["end_time"] = end_time
        
        # Save analysis
        analysis_path = f"output/{meeting_id}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Store in memory
        memory.add_meeting(analysis, meeting_id)
        
        return JSONResponse({
            "status": "success",
            "message": "File processed successfully",
            "meeting_id": meeting_id,
            "analysis": analysis
        })
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    file_path = f"audio/{filename}"
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "Audio file not found"
        }, status_code=404)
    
    return FileResponse(
        path=file_path,
        media_type='audio/mpeg',
        filename=filename
    )

@app.get("/search")
async def search_meetings(query: str, n_results: int = 5):
    """Search through meeting memories."""
    try:
        results = memory.search_meetings(query, n_results)
        return JSONResponse({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error searching meetings: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/meeting/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get meeting history."""
    try:
        results = memory.get_meeting_history(meeting_id)
        return JSONResponse({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error retrieving meeting: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/summary")
async def get_summary():
    """Get summary of all meetings."""
    try:
        summary = memory.summarize_all_meetings()
        return JSONResponse({
            "status": "success",
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/speaker/{speaker_name}")
async def get_speaker_summary(speaker_name: str):
    """Get summary of speaker's contributions."""
    try:
        summary = memory.get_speaker_summary(speaker_name)
        return JSONResponse({
            "status": "success",
            "summary": summary
        })
    except Exception as e:
        logger.error(f"Error generating speaker summary: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 