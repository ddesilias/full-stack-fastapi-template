# Add these imports at the top
import uuid
from typing import Any
import openai
from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, HTTPException
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.models import Item, ItemCreate, ItemPublic, ItemsPublic, ItemUpdate, Message
from app.core.config import settings  # We'll store the API key in settings

router = APIRouter(prefix="/model", tags=["model"])

# Add these new models for the request/response
class ImageAnalysisRequest(BaseModel):
    image_url: HttpUrl
    prompt: str = "Can you describe this image?"

class ImageAnalysisResponse(BaseModel):
    analysis: str

@router.post("/analyze-image", response_model=ImageAnalysisResponse)
def analyze_image(
    *,
    session: SessionDep,
    request: ImageAnalysisRequest
) -> Any:
    """
    Analyze an image using Fireworks AI vision model.
    """
    try:
        client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=settings.FIREWORKS_API_KEY
        )
        
        response = client.chat.completions.create(
            model="accounts/fireworks/models/phi-3-vision-128k-instruct",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": request.prompt,
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": str(request.image_url)
                    },
                }],
            }]
        )
        
        return ImageAnalysisResponse(
            analysis=response.choices[0].message.content
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )