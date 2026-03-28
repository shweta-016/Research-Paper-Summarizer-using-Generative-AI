import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.cleaned_text = ""
        self.sentences = []
        self.words = []

    def extract_text(self):
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                        
        except Exception as e:
            print("Error reading PDF:", e)
        
        self.raw_text = text
        return text

    def remove_references(self, text):
        patterns = ["references", "bibliography", "acknowledgment"]
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            if any(p in line.lower() for p in patterns):
                break
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)

    def remove_special_characters(self, text):
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\([^)]*\)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9., ]', '', text)
        return text

    def to_lowercase(self, text):
        return text.lower()

    def sentence_tokenization(self, text):
        self.sentences = sent_tokenize(text)
        return self.sentences

    def word_tokenization(self, text):
        self.words = word_tokenize(text)
        return self.words

    def remove_stopwords(self, words):
        stop_words = set(stopwords.words("english"))
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return filtered_words

    def stemming(self, words):
        stemmer = nltk.PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in words]
        return stemmed_words

    def preprocessing_pipeline(self):
        print("Extracting text...")
        text = self.extract_text()

        print("Removing references...")
        text = self.remove_references(text)

        print("Cleaning text...")
        text = self.remove_special_characters(text)

        print("Converting to lowercase...")
        text = self.to_lowercase(text)

        print("Tokenizing sentences...")
        sentences = self.sentence_tokenization(text)

        print("Tokenizing words...")
        words = self.word_tokenization(text)

        print("Removing stopwords...")
        filtered_words = self.remove_stopwords(words)

        print("Stemming...")
        stemmed_words = self.stemming(filtered_words)

        self.cleaned_text = " ".join(filtered_words)

        return {
            "cleaned_text": self.cleaned_text,
            "sentences": sentences,
            "words": stemmed_words
        }

    def get_statistics(self):
        stats = {
            "total_sentences": len(self.sentences),
            "total_words": len(self.words),
            "total_characters": len(self.cleaned_text)
        }
        return stats


if __name__ == "__main__":
    processor = PDFProcessor("sample.pdf")
    data = processor.preprocessing_pipeline()
    stats = processor.get_statistics()
    
    print("\nDocument Statistics:")
    print(stats)