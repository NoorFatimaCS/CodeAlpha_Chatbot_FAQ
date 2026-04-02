from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQs
with open('faqs.json', 'r') as f:
    data = json.load(f)

questions = [item['question'] for item in data['queries']]
answers = [item['answer'] for item in data['queries']]

def get_bot_response(user_input):
    vectorizer = TfidfVectorizer()
    all_texts = questions + [user_input]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    match_index = similarities.argmax()
    score = similarities[0][match_index]
    
    if score > 0.3:
        return answers[match_index]
    else:
        return "I'm sorry, I don't have information on that. Please contact support."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    response = get_bot_response(user_query)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)