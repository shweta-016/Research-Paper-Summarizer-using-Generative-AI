from transformers import pipeline

class RAGQA:
    def __init__(self):
        print("Loading QA model...")
        self.model = pipeline("text-generation", model="sshleifer/tiny-gpt2")

    def generate_answer(self, context_chunks, question):
        context = " ".join(context_chunks)
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        result = self.model(prompt, max_length=120, do_sample=False)
        return result[0]['generated_text']

    def answer_with_sources(self, context_chunks, question):
        answer = self.generate_answer(context_chunks, question)
        
        # Show which chunks were used
        sources = []
        for i, chunk in enumerate(context_chunks):
            sources.append(f"Source {i+1}: {chunk[:200]}...")

        return {
            "answer": answer,
            "sources": sources
        }