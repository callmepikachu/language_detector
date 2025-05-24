from lang_detector.detectors.ngram import NGramLanguageDetector
from lang_detector.data.loader import load_tatoeba_data, detect_tatoeba_data
from lang_detector.utils import split_dataset
from lang_detector.schemas import LanguageDetectionRequest


data = load_tatoeba_data(samples_per_lang=1000)
train_data, test_data = split_dataset(data)


X_train = [text for text, _ in train_data]
y_train = [label for _, label in train_data]

detector = NGramLanguageDetector()
detector.fit(X_train, y_train)

correct = 0
for i, (text, label) in enumerate(test_data):
    pred = detector.detect(LanguageDetectionRequest(text=text)).language
    if pred == label:
        correct += 1
    else:
        print(f"[ERROR] index {i}")
        print(f"sentence: {text}")
        print(f"label: {label}")
        print(f"pred: {pred}")
        print("-" * 50)
print(f"Accuracy: {correct/len(test_data):.2%}")

# 调用函数
stats = detect_tatoeba_data(data)

# 打印统计信息
print(f"total sentence: {stats['total']}")
print("\nlang distribution:")
for lang, count in stats['language_counts'].items():
    print(f"{lang}: {count}  ({stats['language_proportions'][lang]:.2%})")
