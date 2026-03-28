import sqlite3
from datetime import datetime

class ResearchDatabase:
    def __init__(self, db_name="research_papers.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            upload_date TEXT
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        )
        """)
        self.conn.commit()

    def insert_paper(self, title, summary):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO papers (title, summary, upload_date) VALUES (?, ?, ?)",
                            (title, summary, date))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_question(self, paper_id, question, answer):
        self.cursor.execute("INSERT INTO questions (paper_id, question, answer) VALUES (?, ?, ?)",
                            (paper_id, question, answer))
        self.conn.commit()

    def get_all_papers(self):
        self.cursor.execute("SELECT * FROM papers")
        return self.cursor.fetchall()

    def get_questions_for_paper(self, paper_id):
        self.cursor.execute("SELECT question, answer FROM questions WHERE paper_id=?", (paper_id,))
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = ResearchDatabase()
    pid = db.insert_paper("AI Paper", "This paper discusses AI.")
    db.insert_question(pid, "What is AI?", "AI is artificial intelligence.")
    print(db.get_all_papers())