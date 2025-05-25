import argparse
from lang_detector.data.loader import load_tatoeba_data, detect_tatoeba_data
from lang_detector.utils import split_dataset
from lang_detector.schemas import LanguageDetectionRequest

from lang_detector.detectors.ngram import NGramLanguageDetector
from lang_detector.detectors.rnn import RNNLanguageDetector

DETECTOR_MAP = {
    'ngram': NGramLanguageDetector,
    'rnn': RNNLanguageDetector,
}


def main():
    parser = argparse.ArgumentParser(description="Language Detection CLI")
    parser.add_argument('--detector', type=str, default="ngram",choices=DETECTOR_MAP.keys(),
                        help='Choose detector: "ngram" or "rnn"')
    parser.add_argument('--samples-per-lang', type=int, default=1000,
                        help='Number of samples per language (default: 1000)')
    args = parser.parse_args()

    # load data
    print(f"[INFO] Loading data with {args.samples_per_lang} samples per language...")
    data = load_tatoeba_data(samples_per_lang=args.samples_per_lang)
    train_data, test_data = split_dataset(data)

    X_train = [text for text, _ in train_data]
    y_train = [label for _, label in train_data]

    # init detector
    DetectorClass = DETECTOR_MAP[args.detector]
    print(f"[INFO] Using detector: {DetectorClass.__name__}")

    if args.detector == 'rnn':
        detector = DetectorClass(num_epochs=10, batch_size=64)
    else:
        detector = DetectorClass()

    # train
    print("[INFO] Training model...")
    detector.fit(X_train, y_train)

    # predict
    print("[INFO] Evaluating model...")
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

    accuracy = correct / len(test_data)
    print(f"Accuracy: {accuracy:.2%}")

    # output the statistics
    stats = detect_tatoeba_data(data)
    print(f"\nTotal sentences: {stats['total']}")
    print("Language distribution:")
    for lang, count in stats['language_counts'].items():
        print(f"{lang}: {count} ({stats['language_proportions'][lang]:.2%})")


if __name__ == '__main__':
    main()