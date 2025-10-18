from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Defines the JSON structure for the analysis response."""
    
    filename: str = Field(description="The name of the uploaded video file.")
    label: str = Field(description="The final ethical classification: 'Safe', 'Suspicious', or 'Harmful'.")
    ethical_score: float = Field(description="A score from 0.0 (Safe) to 1.0 (Harmful).")
    deepfake_probability: float = Field(description="The model's confidence that the video is a deepfake (0.0 to 1.0).")
    harmful_text_probability: float = Field(description="The model's confidence that the text is harmful (0.0 to 1.0).")
    transcript: str | None = Field(None, description="The transcribed text from the video's audio.")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "test_video.mp4",
                "label": "Harmful",
                "ethical_score": 0.91,
                "deepfake_probability": 0.98,
                "harmful_text_probability": 0.85,
                "transcript": "This is example text that was found to be harmful."
            }
        }