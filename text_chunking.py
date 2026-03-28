from typing import List
import math

class TextChunker:
    def __init__(self, chunk_size=300, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    def chunk_by_words(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start = end - self.overlap

        self.chunks = chunks
        return chunks

    def chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            word_count += len(words)
            current_chunk.append(sentence)

            if word_count >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.chunks = chunks
        return chunks

    def add_metadata(self):
        chunk_data = []
        for i, chunk in enumerate(self.chunks):
            chunk_data.append({
                "id": i,
                "text": chunk,
                "length": len(chunk.split())
            })
        return chunk_data

    def get_chunk_statistics(self):
        lengths = [len(chunk.split()) for chunk in self.chunks]
        stats = {
            "total_chunks": len(self.chunks),
            "avg_chunk_size": sum(lengths) / len(lengths) if lengths else 0,
            "max_chunk_size": max(lengths) if lengths else 0,
            "min_chunk_size": min(lengths) if lengths else 0
        }
        return stats


if __name__ == "__main__":
    text = open("sample.txt").read()
    chunker = TextChunker()
    chunks = chunker.chunk_by_words(text)
    stats = chunker.get_chunk_statistics()
    print(stats)