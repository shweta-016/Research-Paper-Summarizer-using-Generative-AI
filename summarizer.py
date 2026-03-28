import nltk
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

class ResearchPaperSummarizer:
    def __init__(self):
        print("Summarizer initialized...")

    def clean_text(self, text):
        # Remove special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9., ]', '', text)

        # Remove repeated words
        words = text.split()
        cleaned_words = []
        for i in range(len(words)):
            if i == 0 or words[i] != words[i-1]:
                cleaned_words.append(words[i])
        return " ".join(cleaned_words)

    def summarize_document(self, chunks, num_sentences=6):
        text = " ".join(chunks)
        text = self.clean_text(text)

        sentences = sent_tokenize(text)

        # TF-IDF scoring
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        scores = np.array(X.sum(axis=1)).flatten()

        # Rank sentences
        ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:]]
        ranked_sentences.sort()

        summary = " ".join(ranked_sentences)
        return summary