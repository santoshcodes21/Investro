from flask import Flask, jsonify
import requests
import logging

app = Flask(__name__)

    # Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/news/<query>')
def get_news(query):
        api_key = '4f25485a971541e9af6be414766541a3'  # Your NewsAPI key
        url = f'https://newsapi.org/v2/everything?q={query}&from=2020-01-01&to=2025-04-30&apiKey={api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.info(f"Fetched news data for query: {query}")
            return jsonify(response.json())
        except requests.exceptions.HTTPError as e:
            logger.error(f"NewsAPI error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e),
                'articles': [
                    {'publishedAt': '2025-04-01T00:00:00Z', 'title': 'Apple reports record-breaking earnings.', 'description': ''},
                    {'publishedAt': '2025-04-03T00:00:00Z', 'title': 'Apple faces supply chain issues.', 'description': ''}
                ]
            }), 200

if __name__ == '__main__':
        app.run(host='localhost', port=5000)