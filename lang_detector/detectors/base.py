from abc import ABC, abstractmethod
from lang_detector.schemas import LanguageDetectionRequest, LanguageDetectionResponse

class BaseLanguageDetector(ABC):
    @abstractmethod
    def detect(self, request: LanguageDetectionRequest) -> LanguageDetectionResponse:
        pass

    @abstractmethod
    def fit(self, texts, labels):
        pass