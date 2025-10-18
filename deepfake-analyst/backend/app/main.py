import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from .schemas import AnalysisResult

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Deepfake Analyst API",
    description="Analyzes video content for deepfakes and ethical concerns.",
    version="1.0.0"
)

# --- 2. Placeholder AI & Processing Functions ---
# These functions simulate the real ML models.
# The ML Engineer will replace the logic inside these.

def extract_media_components(video_path: Path) -> Tuple[bool, str, str]:
    """(ML TASK) Placeholder for extracting frames and audio."""
    print(f"PRETENDING to extract components from {video_path}...")
    # Real logic: Use OpenCV/moviepy to save frames and audio track.
    return (True, "path/to/dummy_frames_folder", "path/to/dummy_audio.wav")

def run_deepfake_detection(frames_path: str) -> float:
    """(ML TASK) Placeholder for the deepfake detection model."""
    print(f"PRETENDING to run deepfake model on frames at {frames_path}...")
    # Real logic: Load XceptionNet model and predict on frames.
    return 0.98  # Dummy result: 98% fake

def transcribe_audio(audio_path: str) -> str:
    """(ML TASK) Placeholder for Whisper audio transcription."""
    print(f"PRETENDING to transcribe audio from {audio_path}...")
    # Real logic: Load Whisper model and transcribe the audio file.
    return "This is example text that was found to be harmful."

def analyze_text_for_harm(transcript: str) -> float:
    """(ML TASK) Placeholder for Hugging Face text analysis."""
    print(f"PRETENDING to analyze text: '{transcript[:30]}...'")
    # Real logic: Use a Hugging Face model for toxicity/sentiment.
    return 0.85  # Dummy result: 85% harmful

def score_and_classify(deepfake_prob: float, harm_prob: float) -> Tuple[str, float]:
    """(BACKEND TASK) Scores and classifies content based on model outputs."""
    # This is a simple scoring rule, can be made more complex.
    ethical_score = (deepfake_prob * 0.6) + (harm_prob * 0.4)

    if ethical_score > 0.75:
        label = "Harmful 🚫"
    elif ethical_score > 0.5:
        label = "Suspicious ⚠"
    else:
        label = "Safe ✅"
        
    return label, round(ethical_score, 2)

# --- 3. The API Endpoint ---

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "Deepfake Analyst API is online."}

@app.post("/api/v1/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_video(video: UploadFile = File(..., description="The video file to be analyzed.")):
    """
    Accepts a video file, runs the full analysis pipeline, and returns the results.
    """
    # Securely save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(video.file, tmp_file)
        tmp_video_path = Path(tmp_file.name)

    try:
        # --- Run the Analysis Pipeline ---
        # Each function is run in a separate thread to keep the API responsive.
        
        # Step 1: Extract frames and audio from the video file.
        success, frames_path, audio_path = await asyncio.to_thread(extract_media_components, tmp_video_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video file.")
            
        # Step 2: Run deepfake detection and audio transcription at the same time.
        deepfake_task = asyncio.to_thread(run_deepfake_detection, frames_path)
        transcribe_task = asyncio.to_thread(transcribe_audio, audio_path)
        deepfake_probability, transcript = await asyncio.gather(deepfake_task, transcribe_task)

        # Step 3: Analyze the transcribed text.
        harmful_text_probability = await asyncio.to_thread(analyze_text_for_harm, transcript)
        
        # Step 4: Calculate the final label and score.
        label, ethical_score = score_and_classify(deepfake_probability, harmful_text_probability)
        
        # Step 5: Return the final, structured result.
        return AnalysisResult(
            filename=video.filename,
            label=label,
            ethical_score=ethical_score,
            deepfake_probability=deepfake_probability,
            harmful_text_probability=harmful_text_probability,
            transcript=transcript
        )
    finally:
        # Clean up by deleting the temporary video file.
        tmp_video_path.unlink()