# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
from improved_nn import NN  # Your neural network class
import os

app = Flask(__name__)

# Initialize the model
# app.py
model = NN()
if os.path.exists('w1.npy') and os.path.exists('means.npy'):
    model.load_model()
else:
    raise FileNotFoundError("Model weights or scaling parameters not found. Please train the model first.")

def extract_features(url):
    """Extract features from a single URL."""
    features = {}
    
    # Check for IP address
    features['use_of_ip'] = 1 if re.match(
        r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
        urlparse(url).netloc
    ) else 0
    
    # Count specific characters
    features['count.'] = url.count('.')
    features['@ Precence'] = 1 if '@' in url else 0
    features['- Precence'] = 1 if '-' in url else 0
    features['∼ Precence'] = 1 if '~' in url else 0
    
    # Count embedded domains
    features['count_embed_domian'] = len(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', url))
    
    # Check for suspicious TLD
    suspicious_tlds = ['.zip', '.review', '.country', '.kim', '.cricket', '.science', '.work', '.party', '.gq', '.link']
    features['sus_url'] = 1 if any(tld in url.lower() for tld in suspicious_tlds) else 0
    
    # Check for URL shortening services
    shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'is.gd']
    features['short_url'] = 1 if any(service in url.lower() for service in shortening_services) else 0
    
    # Check HTTPS in domain
    features['HTTPS in Domain'] = 1 if 'https' in urlparse(url).netloc.lower() else 0
    
    # URL length
    features['url_length'] = len(url)
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_url():
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url']
        
        # Extract features
        features = extract_features(url)
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[[
            'use_of_ip', 'count.', '@ Precence', '- Precence', '∼ Precence',
            'count_embed_domian', 'sus_url', 'short_url', 'HTTPS in Domain', 'url_length'
        ]]
        
        # Make prediction
        prediction = model.predict(feature_df)
        
        # Since 0 means malicious and 1 means safe in your data,
        # we need to invert the prediction to get the correct is_malicious value
        result = {
            'url': url,
            'is_malicious': bool(prediction[0][0] == 0),  # Added bool() conversion
            'features': features
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({'error': 'No URLs provided'}), 400
        
        urls = data['urls']
        results = []
        
        for url in urls:
            features = extract_features(url)
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[[
                'use_of_ip', 'count.', '@ Precence', '- Precence', '∼ Precence',
                'count_embed_domian', 'sus_url', 'short_url', 'HTTPS in Domain', 'url_length'
            ]]
            
            prediction = model.predict(feature_df)
            
            results.append({
            'url': url,
            'is_malicious': bool(prediction[0][0] == 0),  # Added bool() conversion
            'features': features
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)