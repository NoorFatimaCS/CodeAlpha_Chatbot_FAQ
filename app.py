from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer

# Purane try-except ko hata kar sirf ye 3 lines likhen
nltk.download('punkt_tab') # Punkt ka naya version
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# Load FAQs
with open('faqs.json', 'r') as f:
    data = json.load(f)

questions = [item['question'] for item in data['queries']]
answers = [item['answer'] for item in data['queries']]

# --- UPDATED: Professional get_bot_response function ---
def get_bot_response(user_input):
    # 1. Text Cleaning: User ki input ko process karna (e.g., "submitting" becomes "submit")
    tokens = nltk.word_tokenize(user_input.lower())
    clean_input = " ".join([lemmatizer.lemmatize(w) for w in tokens])
    
    # 2. Vectorization
    vectorizer = TfidfVectorizer()
    # Hum questions aur cleaned input dono ko vectorizer mein daalenge
    all_texts = questions + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # 3. Similarity Check
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    match_index = similarities.argmax()
    score = similarities[0][match_index]

    if score > 0.3:
        return answers[match_index], score
    else:
        # Professional fallback message
        fallback = "I'm sorry, I don't have information on that. You can contact support at services@codealpha.tech."
        return fallback, score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    answer, score = get_bot_response(user_query)

    # Logs save karna
    with open("logs.txt", "a") as f:
        f.write(f"Query: {user_query} | Score: {score}\n")

    return jsonify({
        "answer": answer,
        "score": float(score)
    })

if __name__ == '__main__':
    # Hugging Face deployment ke liye host aur port settings
    app.run(host='0.0.0.0', port=7860)
