import unittest
from lang_detector.detectors.ngram import NGramLanguageDetector
from lang_detector.schemas import LanguageDetectionRequest

class TestNGramDetector(unittest.TestCase):
    def setUp(self):
        self.detector = NGramLanguageDetector()
        self.detector.fit(
            ["hello world", "bonjour le monde", "hola mundo", "你好世界"],
            ["en", "fr", "es", "zh"]
        )

    def test_detect(self):
        self.assertEqual(self.detector.detect(LanguageDetectionRequest(text="hello")).language, "en")
        self.assertEqual(self.detector.detect(LanguageDetectionRequest(text="bonjour")).language, "fr")
        self.assertEqual(self.detector.detect(LanguageDetectionRequest(text="hola")).language, "es")
        self.assertEqual(self.detector.detect(LanguageDetectionRequest(text="你好")).language, "zh")

if __name__ == '__main__':
    unittest.main()