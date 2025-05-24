from pydantic import BaseModel
from typing import Optional, List

class LanguageDetectionRequest(BaseModel):
    text: str

class LanguageDetectionResponse(BaseModel):
    language: str
    confidence: float = 1.0
    possible_languages: Optional[List[str]] = None