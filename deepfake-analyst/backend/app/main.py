import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

# --- Core FastAPI & Utils ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AnalysisResult
import os

# --- ML Model Imports ---
from moviepy import VideoFileClip
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
import cv2
import numpy as np
import whisper
from transformers import pipeline
import torch # Needed for text analysis pipeline if it defaults to PyTorch

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Deepfake Analyst API",
    description="Analyzes video content for deepfakes and ethical concerns.",
    version="1.0.0"
)

# --- 2. Add CORS Middleware ---
origins = ["*"]  # Allow all origins (good for a hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Global Variables for ML Models ---
# Tracking two deepfake models for ensemble
DEEPFAKE_MODEL_CNN = None
DEEPFAKE_MODEL_MESO4 = None
CNN_INPUT_SIZE = (256, 256) # Default, will update
MESO4_INPUT_SIZE = (256, 256) # Meso4 uses 256x256
# Other models
WHISPER_MODEL = None
TEXT_ANALYSIS_PIPELINE = None

# --- 4. Model Definition & Loading Functions ---

# --- ML STEP 1: VIDEO EXTRACTION (Unchanged) ---
def extract_media_components(video_path: Path, output_dir_path: str) -> Tuple[bool, str, str]:
    print(f"Extracting components into {output_dir_path}...")
    try:
        audio_file_path = os.path.join(output_dir_path, "audio.wav")
        frames_dir_path = os.path.join(output_dir_path, "frames")
        os.makedirs(frames_dir_path)
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio:
                clip.audio.write_audiofile(audio_file_path, codec='pcm_s16le', logger=None)
            else:
                audio_file_path = None
            frames_pattern = os.path.join(frames_dir_path, "frame_%04d.png")
            clip.write_images_sequence(frames_pattern, logger=None)
        print(f"Extraction complete.")
        return (True, frames_dir_path, audio_file_path)
    except Exception as e:
        print(f"Error during media extraction: {e}")
        return (False, None, None)

# --- ML STEP 2: DEEPFAKE DETECTION (ENSEMBLE) ---

# --- MESO4 SKELETON DEFINITION ---
def Meso4():
    x = Input(shape = (256, 256, 3))
    x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    y = Flatten()(x4)
    y = layers.Dropout(0.5)(y)
    y = Dense(16)(y)
    try: # Handle potential Keras version differences in LeakyReLU
        y = layers.LeakyReLU(negative_slope=0.1)(y)
    except TypeError:
        y = layers.LeakyReLU(alpha=0.1)(y) # Fallback for older Keras versions
    y = layers.Dropout(0.5)(y)
    y = Dense(1, activation = 'sigmoid')(y)
    return Model(inputs = x, outputs = y)

def load_deepfake_models(): # Renamed to plural
    """Loads BOTH deepfake models on startup."""
    global DEEPFAKE_MODEL_CNN, DEEPFAKE_MODEL_MESO4, CNN_INPUT_SIZE, MESO4_INPUT_SIZE

    # --- Load CNN Model ---
    try:
        cnn_model_path = "backend/models/cnn_model.h5"
        DEEPFAKE_MODEL_CNN = tf.keras.models.load_model(cnn_model_path)
        print(f"CNN Deepfake model loaded successfully from {cnn_model_path}")
        input_shape = DEEPFAKE_MODEL_CNN.input_shape
        print(f"*** CNN Model expects input shape: {input_shape} ***")
        if len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None:
             CNN_INPUT_SIZE = (input_shape[1], input_shape[2])
             print(f"*** Updated CNN_INPUT_SIZE to: {CNN_INPUT_SIZE} ***")
        else:
             print(f"*** Could not determine CNN input size, using default: {CNN_INPUT_SIZE} ***")
        DEEPFAKE_MODEL_CNN.predict(np.zeros((1, CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 3)))
        print("CNN Model is warmed up.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load CNN TensorFlow model: {e}")
        DEEPFAKE_MODEL_CNN = None

    # --- Load Meso4 Model ---
    try:
        meso4_model_path = "backend/models/Meso4_DF.h5"
        DEEPFAKE_MODEL_MESO4 = Meso4()
        DEEPFAKE_MODEL_MESO4.load_weights(meso4_model_path)
        print(f"Meso4 Deepfake model loaded successfully from {meso4_model_path}")
        MESO4_INPUT_SIZE = (256, 256) # Meso4 is fixed at 256x256
        DEEPFAKE_MODEL_MESO4.predict(np.zeros((1, MESO4_INPUT_SIZE[0], MESO4_INPUT_SIZE[1], 3)))
        print("Meso4 Model is warmed up.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load Meso4 TensorFlow model: {e}")
        DEEPFAKE_MODEL_MESO4 = None

def run_deepfake_detection(frames_path: str) -> float:
    """(ML TASK 2) Runs BOTH deepfake models and averages the scores."""
    global DEEPFAKE_MODEL_CNN, DEEPFAKE_MODEL_MESO4, CNN_INPUT_SIZE, MESO4_INPUT_SIZE

    if DEEPFAKE_MODEL_CNN is None and DEEPFAKE_MODEL_MESO4 is None:
        print("ERROR: No deepfake models loaded successfully.")
        return 0.5 # Return neutral if both failed

    print(f"Running Ensemble deepfake detection on frames in {frames_path}...")
    try:
        frame_files = [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.png')]
        sample_frames = np.random.choice(frame_files, min(len(frame_files), 10), replace=False)

        cnn_predictions = []
        meso4_predictions = []

        for frame_file in sample_frames:
            img = cv2.imread(frame_file)

            # --- Predict with CNN Model (if loaded) ---
            if DEEPFAKE_MODEL_CNN is not None:
                img_resized_cnn = cv2.resize(img, CNN_INPUT_SIZE)
                img_normalized_cnn = img_resized_cnn / 255.0
                img_batch_cnn = np.expand_dims(img_normalized_cnn, axis=0)
                pred_cnn = DEEPFAKE_MODEL_CNN.predict(img_batch_cnn, verbose=0)[0][0]
                cnn_predictions.append(pred_cnn)

            # --- Predict with Meso4 Model (if loaded) ---
            if DEEPFAKE_MODEL_MESO4 is not None:
                img_resized_meso4 = cv2.resize(img, MESO4_INPUT_SIZE)
                img_normalized_meso4 = img_resized_meso4 / 255.0
                img_batch_meso4 = np.expand_dims(img_normalized_meso4, axis=0)
                pred_meso4 = DEEPFAKE_MODEL_MESO4.predict(img_batch_meso4, verbose=0)[0][0]
                meso4_predictions.append(pred_meso4)

        # --- Combine Scores ---
        final_score_cnn = np.mean(cnn_predictions) if cnn_predictions else None
        final_score_meso4 = np.mean(meso4_predictions) if meso4_predictions else None

        print(f"CNN Score: {final_score_cnn}, Meso4 Score: {final_score_meso4}")

        # Simple Averaging (handle cases where one model failed to load)
        if final_score_cnn is not None and final_score_meso4 is not None:
            final_score = (final_score_cnn + final_score_meso4) / 2.0
        elif final_score_cnn is not None:
            final_score = final_score_cnn # Only CNN worked
        elif final_score_meso4 is not None:
            final_score = final_score_meso4 # Only Meso4 worked
        else:
            final_score = 0.0 # Should not happen if check at start passed

        print(f"Ensemble Deepfake detection score: {final_score}")
        return float(final_score)

    except Exception as e:
        print(f"Error during deepfake detection: {e}")
        return 0.5

# --- ML STEP 3: AUDIO TRANSCRIPTION (Unchanged) ---
def load_whisper_model():
    global WHISPER_MODEL
    try:
        WHISPER_MODEL = whisper.load_model("tiny")
        print("Whisper model 'tiny' loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load Whisper model: {e}")
        WHISPER_MODEL = None

def transcribe_audio(audio_path: str) -> str:
    global WHISPER_MODEL
    if audio_path is None: return ""
    if WHISPER_MODEL is None: return "[Whisper model failed to load]"
    print(f"Transcribing audio from {audio_path}...")
    try:
        result = WHISPER_MODEL.transcribe(audio_path)
        transcript = result.get("text", "")
        print(f"Transcription complete: '{transcript[:40]}...'")
        return transcript
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return "[Error during transcription]"

# --- ML STEP 4: TEXT ANALYSIS (Unchanged) ---
def load_text_pipeline():
    global TEXT_ANALYSIS_PIPELINE
    try:
        model_name = "unitary/toxic-bert"
        TEXT_ANALYSIS_PIPELINE = pipeline("text-classification", model=model_name, tokenizer=model_name)
        print(f"Hugging Face pipeline '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load text analysis pipeline: {e}")
        TEXT_ANALYSIS_PIPELINE = None

def analyze_text_for_harm(transcript: str) -> float:
    global TEXT_ANALYSIS_PIPELINE
    if not transcript: return 0.0
    if TEXT_ANALYSIS_PIPELINE is None: return 0.5
    print(f"Analyzing text: '{transcript[:40]}...'")
    try:
        results = TEXT_ANALYSIS_PIPELINE(transcript, truncation=True)
        harm_score = 0.0
        for result in results:
             label = result.get('label', '').upper()
             score = result.get('score', 0.0)
             if label == 'TOXIC':
                harm_score = score
                break
        print(f"Harmful text score: {harm_score}")
        return float(harm_score)
    except Exception as e:
        print(f"Error during text analysis: {e}")
        return 0.5

# --- 5. BACKEND SCORING FUNCTION (Threshold needs careful testing!) ---
def score_and_classify(deepfake_prob: float, harm_prob: float) -> Tuple[str, float]:
    # !!! You MUST test and adjust this threshold based on the ENSEMBLE scores !!!
    DEEPFAKE_THRESHOLD = 0.80 # Starting point - RE-EVALUATE
    # !!!

    is_fake = deepfake_prob > DEEPFAKE_THRESHOLD
    is_harmful = harm_prob > 0.7

    if is_fake and is_harmful:
        label = "Harmful 🚫"
    elif is_fake and not is_harmful:
        label = "Suspicious ⚠"
    else: # If not fake enough (below threshold)
        if is_harmful:
             label = "Suspicious ⚠" # Real video but potentially harmful text? Flag it.
        else:
             label = "Safe ✅" # Real video, harmless text.

    ethical_score = (deepfake_prob * 0.6) + (harm_prob * 0.4)
    return label, round(ethical_score, 2)

# --- 6. SERVER STARTUP EVENT (UPDATED) ---
@app.on_event("startup")
async def startup_event():
    """Loads all ML models when the server starts."""
    print("--- Server starting up, loading all models... ---")
    load_deepfake_models() # <-- Calls the function that loads BOTH deepfake models
    load_whisper_model()
    load_text_pipeline()
    print("--- All models loaded (or attempted). Server is ready. ---")

# --- 7. The API Endpoint (Unchanged) ---
@app.get("/")
def read_root():
    return {"status": "Deepfake Analyst API is online."}

@app.post("/api/v1/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_video(video: UploadFile = File(..., description="The video file to be analyzed.")):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
        shutil.copyfileobj(video.file, tmp_video_file)
        tmp_video_path = Path(tmp_video_file.name)

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        try:
            success, frames_path, audio_path = await asyncio.to_thread(
                extract_media_components, tmp_video_path, tmp_output_dir
            )
            if not success:
                raise HTTPException(status_code=500, detail="Failed to process video file.")

            deepfake_task = asyncio.to_thread(run_deepfake_detection, frames_path) # Now runs ensemble
            transcribe_task = asyncio.to_thread(transcribe_audio, audio_path)
            deepfake_probability, transcript = await asyncio.gather(deepfake_task, transcribe_task)

            harmful_text_probability = await asyncio.to_thread(analyze_text_for_harm, transcript)

            label, ethical_score = score_and_classify(deepfake_probability, harmful_text_probability)

            return AnalysisResult(
                filename=video.filename,
                label=label,
                ethical_score=ethical_score,
                deepfake_probability=deepfake_probability, # This is now the ENSEMBLE score
                harmful_text_probability=harmful_text_probability,
                transcript=transcript
            )
        finally:
            tmp_video_path.unlink()