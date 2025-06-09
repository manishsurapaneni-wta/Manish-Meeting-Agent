# Meeting Copilot

An AI-powered meeting assistant that transcribes audio, analyzes content, and generates comprehensive meeting summaries using WhisperX and CrewAI.

## Features

- 🎙️ High-quality audio transcription with speaker diarization using WhisperX
- 🤖 Multi-agent analysis using CrewAI and GPT-4
- 📝 Meeting summarization
- ✅ Decision extraction
- 📋 Action item tracking
- 🔄 Follow-up identification

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg installed on your system
- Hugging Face account and API token (for speaker diarization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meeting_copilot.git
cd meeting_copilot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

To get your Hugging Face token:
1. Create an account at https://huggingface.co/
2. Go to Settings -> Access Tokens
3. Create a new token with read access
4. Copy the token to your `.env` file

## Usage

1. Place your meeting audio file in the `audio/` directory

2. Run the pipeline:
```bash
python app.py --audio_file your_meeting.mp3
```

3. Find the analysis results in `output/meeting_summary.json`

## Testing

To test the transcription functionality:

1. Place a test audio file in the `audio/` directory
2. Run the test script:
```bash
python scripts/test_transcription.py audio/test_meeting.mp3
```

The script will:
- Transcribe the audio with speaker diarization
- Save the raw transcription to `output/test_transcription_raw.json`
- Print sample segments and statistics

## Project Structure

```
meeting_copilot/
├── audio/          # Uploaded meeting audio files
├── agents/         # CrewAI agent definitions
├── scripts/        # Whisper and pipeline logic
├── output/         # Final report
├── app.py          # Entrypoint
├── requirements.txt
└── README.md
```

## License

MIT License 

## Docker Setup

To use the Docker setup:

1. **Build the image**:
   ```bash
   docker build -t meeting-copilot .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -v $(pwd)/audio:/app/audio \
     -v $(pwd)/output:/app/output \
     -e OPENAI_API_KEY=your_key \
     -e HF_TOKEN=your_token \
     meeting-copilot
   ```

3. **Access the Application**:
   - Open http://localhost:8000 in your browser
   - The application will be running in a container
   - Audio files and outputs will persist in your local directories

**Important Notes**:

1. **Environment Variables**:
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_openai_api_key
     HF_TOKEN=your_huggingface_token
     ```
   - Docker Compose will automatically use these variables

2. **Data Persistence**:
   - Audio files are stored in `./audio`
   - Output files are stored in `./output`
   - These directories are mounted as volumes

3. **Security**:
   - API keys are passed through environment variables
   - Sensitive files are excluded via .dockerignore
   - The container runs with minimal privileges

4. **Performance**:
   - Uses Python slim image for smaller size
   - Implements Docker layer caching
   - Excludes unnecessary files from build 