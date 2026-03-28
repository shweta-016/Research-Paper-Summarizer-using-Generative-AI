from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class EvaluationMetrics:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_rouge(self, reference, generated):
        scores = self.rouge.score(reference, generated)
        return scores

    def semantic_similarity(self, text1, text2):
        emb1 = self.embedder.encode([text1])
        emb2 = self.embedder.encode([text2])
        similarity = cosine_similarity(emb1, emb2)
        return similarity[0][0]

    def compression_ratio(self, original, summary):
        return len(summary.split()) / len(original.split())

    def evaluate_summary(self, original, summary):
        rouge_scores = self.calculate_rouge(original, summary)
        similarity = self.semantic_similarity(original, summary)
        compression = self.compression_ratio(original, summary)

        return {
            "ROUGE": rouge_scores,
            "Semantic Similarity": similarity,
            "Compression Ratio": compression
        }


if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    original = "Artificial intelligence is a field of computer science."
    summary = "AI is a branch of computer science."
    print(evaluator.evaluate_summary(original, summary))