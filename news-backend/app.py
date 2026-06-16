from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
# CORS backend fetch requests handling-kaga use panrom
CORS(app) 

print("Loading Trained AI Model weights...")
# Namba train panna model files-ah load panrom
model = joblib.load('news_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/classify', methods=['POST'])
def classify_news():
    try:
        req_data = request.get_json()
        articles = req_data.get('articles', [])
        
        processed_articles = []
        for article in articles:
            title_and_desc = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Text-ah matrix vectors-ah mathurathu
            vector_text = vectorizer.transform([title_and_desc])
            prediction = model.predict(vector_text)[0] # 'positive', 'negative' or 'neutral'
            
            # Map predictions to frontend keys
            if prediction == 'positive':
                ai_category = 'good news'
            elif prediction == 'negative':
                ai_category = 'bad news'
            else:
                ai_category = 'neutral'
                
            article['ai_category'] = ai_category
            processed_articles.append(article)
            
        return jsonify({"status": "success", "articles": processed_articles})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Local host port 5000 setup
    app.run(port=5000, debug=True)