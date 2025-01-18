# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:10:39 2025

@author: User
"""

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import requests
from bs4 import BeautifulSoup
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')  # Κατεβάζει τα απαραίτητα δεδομένα για tokenization
nltk.download('stopwords')  # Κατεβάζει την λίστα με τις stopwords

class WikipediaScraper:
    """Class για συλλογή δεδομένων από την Wikipedia."""
    
    def __init__(self, base_url, csv_file):
        # Ορίζει τη βασική διεύθυνση URL και το όνομα του αρχείου CSV για αποθήκευση δεδομένων
        self.base_url = base_url
        self.csv_file = csv_file

    def scrape(self, topic_list):
        """Συλλέγει άρθρα από τη Wikipedia βάσει των θεμάτων και τα αποθηκεύει σε CSV."""
        if os.path.exists(self.csv_file):
            # Αν υπάρχει ήδη το αρχείο, παραλείπεται η συλλογή δεδομένων
            print(f"File {self.csv_file} already exists. Skipping scraping.")
            return

        with open(self.csv_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['doc_id', 'content'])  # Γράφει την κεφαλίδα
            doc_id = 1

            for topic in topic_list:
                # Δημιουργεί το URL για κάθε θέμα
                url = f"{self.base_url}{topic}"
                print(f"Scraping: {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    # Χρησιμοποιεί το BeautifulSoup για να εξάγει το περιεχόμενο των παραγράφων
                    soup = BeautifulSoup(response.content, 'html.parser')
                    paragraphs = soup.find_all('p')
                    content = " ".join([para.get_text() for para in paragraphs])
                    if content.strip():  # Αποθηκεύει μόνο αν το περιεχόμενο δεν είναι κενό
                        writer.writerow([doc_id, content])
                        doc_id += 1
                else:
                    print(f"Failed to retrieve: {url}")

        print(f"Scraping complete. Data saved to {self.csv_file}")


class SearchEngine:
    """Μηχανή αναζήτησης για επεξεργασία ερωτημάτων και επιστροφή αποτελεσμάτων."""
    
    def __init__(self, csv_file):
        # Ορίζει το αρχείο CSV, το dictionary εγγράφων και το inverted index
        self.csv_file = csv_file
        self.documents = {}
        self.inverted_index = defaultdict(list)
        self.stop_words = set(stopwords.words('english'))  # Φορτώνει τις stopwords
        self.stemmer = PorterStemmer()  # Ορίζει τον stemmer
        self.lemmatizer = WordNetLemmatizer()  # Ορίζει τον lemmatizer
        self._load_documents()  # Φορτώνει τα έγγραφα από το CSV
        self._build_inverted_index()  # Δημιουργεί το inverted index

    def _load_documents(self):
        """Φορτώνει έγγραφα από το αρχείο CSV."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"File {self.csv_file} does not exist.")

        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Παραλείπει την κεφαλίδα
            for row in reader:
                doc_id, content = row[0], row[1]
                self.documents[doc_id] = content

    def _preprocess(self, text):
        """Επεξεργάζεται και κανονικοποιεί το κείμενο."""
        # Αφαιρεί ειδικούς χαρακτήρες
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        tokens = word_tokenize(text.lower())  # Μετατρέπει σε μικρά γράμματα και δημιουργεί tokens
        tokens = [token for token in tokens if token not in self.stop_words and token not in string.punctuation]  # Φιλτράρει stopwords και σημεία στίξης
        tokens = [self.stemmer.stem(self.lemmatizer.lemmatize(token)) for token in tokens]  # Εφαρμόζει stemming και lemmatization
        return tokens

    def _build_inverted_index(self):
        """Δημιουργεί το inverted index από τα έγγραφα."""
        for doc_id, content in self.documents.items():
            tokens = self._preprocess(content)
            for token in set(tokens):
                self.inverted_index[token].append(doc_id)  # Προσθέτει τα έγγραφα όπου εμφανίζεται το token

    def search(self, query):
        """Αναζητά έγγραφα που ταιριάζουν με το ερώτημα."""
        query_tokens = self._preprocess(query)  # Επεξεργάζεται το ερώτημα
        if not query_tokens:
            return []

        doc_scores = defaultdict(int)  # Αποθηκεύει τη συχνότητα των tokens ανά έγγραφο
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_id in self.inverted_index[token]:
                    doc_scores[doc_id] += 1

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)  # Ταξινομεί τα έγγραφα με βάση τις βαθμολογίες
        return [doc_id for doc_id, _ in sorted_docs]  # Επιστρέφει τα doc_ids των εγγράφων

    def evaluate(self, test_queries, test_labels):
        """Αξιολογεί τη μηχανή αναζήτησης με precision, recall, και F1-score."""
        y_true = []
        y_pred = []

        for query, relevant_docs in zip(test_queries, test_labels):
            retrieved_docs = self.search(query)  # Αναζητά τα έγγραφα για το ερώτημα
            y_true.extend([1 if doc in relevant_docs else 0 for doc in self.documents.keys()])
            y_pred.extend([1 if doc in retrieved_docs else 0 for doc in self.documents.keys()])

        # Υπολογίζει τα μέτρα αξιολόγησης
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return precision, recall, f1


if __name__ == "__main__":
    # Βήμα 1: Ο χρήστης εισάγει θέματα
    print("Enter Wikipedia topics separated by commas (e.g., Natural_language_processing,Python_(programming_language)):")
    user_input = input("Topics: ")
    topics = [topic.strip() for topic in user_input.split(',') if topic.strip()]  # Διαχωρίζει και καθαρίζει τα θέματα

    if not topics:
        print("No topics provided. Exiting.")
        exit()

    # Βήμα 2: Συλλογή δεδομένων από τη Wikipedia
    scraper = WikipediaScraper("https://en.wikipedia.org/wiki/", "wikipedia_documents.csv")
    scraper.scrape(topics)

    # Βήμα 3: Εκκίνηση της μηχανής αναζήτησης
    search_engine = SearchEngine('wikipedia_documents.csv')

    # Αξιολόγηση (προαιρετικά)
    test_queries = ["machine learning", "neural networks"]
    test_labels = [["1", "2"], ["3", "4"]]  # Αντικατάσταση με πραγματικά doc IDs
    precision, recall, f1 = search_engine.evaluate(test_queries, test_labels)
    print(f"Evaluation Results - Precision: {precision}, Recall: {recall}, F1-score: {f1}")

    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = search_engine.search(query)
        if results:
            print("Documents found:")
            for doc_id in results:
                print(f"- {doc_id}: {search_engine.documents[doc_id][:1000]}...")  # Εμφανίζει τα πρώτα 1000 χαρακτήρες
        else:
            print("No documents found.")