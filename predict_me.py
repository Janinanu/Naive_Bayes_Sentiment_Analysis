import argparse
import pickle
import math
import os


class Classifier:

    def __init__(self, model_file):

            with open(model_file, "rb") as model:
                self.word_probs, self.rel_freqs_pos, self.rel_freqs_neg = pickle.load(model)
            self.backoff_pos = 1 - sum(self.rel_freqs_pos.values())
            self.backoff_neg = 1 - sum(self.rel_freqs_neg.values())

    def classify_word(self, word, c):
        if word in self.word_probs:
            if c == "pos":
                value = self.backoff_pos*self.word_probs[word]
                return self.rel_freqs_pos[word] + value if word in self.rel_freqs_pos else value
            if c == "neg":
                value = self.backoff_neg*self.word_probs[word]
                return self.rel_freqs_neg[word] + value if word in self.rel_freqs_neg else value
        return 1

    def classify_review(self, review_data):
        with open(review_data, "r") as review:
            prob_pos = 0
            prob_neg = 0
            for line in review:
                for word in line.strip().split():
                    prob_pos += math.log(self.classify_word(word, "pos"))
                    prob_neg += math.log(self.classify_word(word, "neg"))
            return "positive" if prob_pos > prob_neg else "negative"

    def classify(self, test_dir):
        correct = 0
        wrong = 0
        with open("classifications.txt", "w") as out:
            for root, _, files in os.walk(test_dir):
                for file in files:
                    pred = self.classify_review(os.path.join(root, file))
                    out.write(pred + "\n")
                    if (root.endswith("pos") and pred == "positive") \
                            or (root.endswith("neg") and pred == "negative"):
                            correct += 1
                    else: wrong += 1
            accuracy = correct/(correct+wrong)
            print(accuracy)

parser = argparse.ArgumentParser(
    description="Train Naive Bayes Model for sentiment classification")
parser.add_argument("model", type=str, help="saved tables")
parser.add_argument("test_dir", type=str, help="directory containing files to predict")
args = parser.parse_args()

classifier = Classifier(args.model)
classifier.classify(args.test_dir)
