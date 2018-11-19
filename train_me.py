import argparse
import os
from collections import Counter
import pickle


class Trainer:

    def __init__(self, train_dir, stopwords_file):

        self.word_freqs_pos = Counter()
        self.word_freqs_neg = Counter()

        with open(stopwords_file) as stopwords:
            stopwords = [word for line in stopwords for word in line.strip().split(",")]
        self.ignore = [".", ",", ":", ";", "(", ")", "[", "]", "?", "!", "-", "'", "\"", "\\", "/", "*"] + stopwords

        for (dirpath, _, filenames) in os.walk(train_dir):

                if dirpath.endswith('pos'):
                    self.count_freqs(dirpath, filenames, self.word_freqs_pos)

                elif dirpath.endswith('neg'):
                    self.count_freqs(dirpath, filenames, self.word_freqs_neg)

    def count_freqs(self, dirpath, filenames, counter):

        for file in filenames:
            with open(os.sep.join([dirpath, file])) as f:
                counter.update([word for line in f for word in line.strip().split() if word not in self.ignore])

    def train(self, model_file):

        word_freqs_total = self.word_freqs_pos + self.word_freqs_neg

        n1 = sum([1 for count in word_freqs_total.values() if count == 1])
        n2 = sum([1 for count in word_freqs_total.values() if count == 2])

        discount = n1/(n1+2*n2)

        freq_pos = sum(self.word_freqs_pos.values())
        freq_neg = sum(self.word_freqs_neg.values())

        rel_freqs_pos = {word : (freq - discount)/freq_pos for word, freq in self.word_freqs_pos.items()}
        rel_freqs_neg = {word : (freq - discount)/freq_neg for word, freq in self.word_freqs_neg.items()}

        word_probs = {word :  freq/(freq_pos+freq_neg) for word, freq in word_freqs_total.items()}

        with open(model_file, "wb") as out:
            pickle.dump((word_probs, rel_freqs_pos,
                         rel_freqs_neg), out)


parser = argparse.ArgumentParser(
    description="Training of Naive Bayes Model for sentiment classification")
parser.add_argument(
    "train_dir", type=str, help="directory with positive and negative training data in sub-folders")
parser.add_argument("stopwords", type=str, help="file of words to ignore")
parser.add_argument("model", type=str,
                    help="file to save the trained model")
args = parser.parse_args()

naive_bayes = Trainer(args.train_dir, args.stopwords)
naive_bayes.train(args.model)

