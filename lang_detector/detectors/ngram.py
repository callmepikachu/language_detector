from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from lang_detector.detectors.base import BaseLanguageDetector
from lang_detector.schemas import LanguageDetectionRequest, LanguageDetectionResponse
import numpy as np

class NGramLanguageDetector(BaseLanguageDetector):
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
        self.classifier = MultinomialNB()
        self.classes_ = []

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = self.classifier.classes_

    def detect(self, request: LanguageDetectionRequest) -> LanguageDetectionResponse:
        X_test = self.vectorizer.transform([request.text])
        proba = self.classifier.predict_proba(X_test)[0]
        lang = self.classifier.predict(X_test)[0]
        confidence = np.max(proba)
        return LanguageDetectionResponse(
            language=lang,
            confidence=confidence,
            possible_languages=[
                self.classes_[i] for i in np.argsort(proba)[::-1] if proba[i] > 0.01
            ]
        )