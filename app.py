from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import mysql.connector
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Create Flask app
app = Flask(__name__)

# Load saved ML model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def get_stats():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root@425",
        database="fakenews_dw"
    )
    cursor = conn.cursor()
    
    # Total articles
    cursor.execute("SELECT COUNT(*) FROM FactNews")
    total = cursor.fetchone()[0]
    
    # Fake count
    cursor.execute("SELECT COUNT(*) FROM FactNews WHERE label_id=0")
    fake = cursor.fetchone()[0]
    
    # Real count
    cursor.execute("SELECT COUNT(*) FROM FactNews WHERE label_id=1")
    real = cursor.fetchone()[0]
    
    # Top categories
    cursor.execute("""
        SELECT c.category_name, COUNT(*) as cnt
        FROM FactNews f
        JOIN DimCategory c ON f.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY cnt DESC
        LIMIT 5
    """)
    categories = cursor.fetchall()
    
    conn.close()
    return total, fake, real, categories

@app.route('/')
def home():
    total, fake, real, categories = get_stats()
    return render_template('index.html',
        total=total,
        fake=fake,
        real=real,
        categories=categories
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from user
    text = request.form.get('news_text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'})
    
    # Clean the text
    cleaned = clean_text(text)
    
    # Convert to numbers using vectorizer
    vectorized = vectorizer.transform([cleaned])
    
    # Predict using ML model
    prediction = model.predict(vectorized)[0]
    
    # Get confidence percentage
    confidence = max(model.predict_proba(vectorized)[0]) * 100
    
    label = 'REAL' if prediction == 1 else 'FAKE'
    
    return jsonify({
        'label': label,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)