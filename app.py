from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
from improved_nn import NN 
import os

app = Flask(__name__)

# Initialize the model
# app.py
model = NN()
model.load_model()

def extract_features(url):
    
    # Check for IP address
    ip_regex = r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|(0x[0-9a-fA-F]{1,2}\.[0x[0-9a-fA-F]{1,2}\.[0x[0-9a-fA-F]{1,2}\.[0x[0-9a-fA-F]{1,2]})'

    # Count specific characters
    dot_regex = r'\.'
    at_regex = r'@'
    dash_regex = r'\-'
    tilde_regex = r'~'

    # Count embedded domains
    embed_domain_regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    # Check for suspicious TLDs
    suspicious_tlds = ['.zip', '.review', '.country', '.kim', '.cricket', '.science', '.work', '.party', '.gq', '.link']

    # Check for URL shortening services
    shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'is.gd']
    features = {}

    # Check for IP address
    features['use_of_ip'] = 1 if re.search(ip_regex, url) else 0

    # Count specific characters
    features['count.'] = len(re.findall(dot_regex, url))
    features['@ Precence'] = 1 if re.search(at_regex, url) else 0
    features['- Precence'] = 1 if re.search(dash_regex, url) else 0
    features['∼ Precence'] = 1 if re.search(tilde_regex, url) else 0

    # Count embedded domains
    features['count_embed_domian'] = len(re.findall(embed_domain_regex, url))

    # Check for suspicious TLD
    features['sus_url'] = 1 if any(tld in url.lower() for tld in suspicious_tlds) else 0

    # Check for URL shortening services
    features['short_url'] = 1 if any(service in url.lower() for service in shortening_services) else 0

    # Check HTTPS in domain
    features['HTTPS in Domain'] = 1 if urlparse(url).scheme.lower() == 'https' else 0

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
            'is_malicious': bool(prediction[0][0] == 1),  # Added bool() conversion
            'features': features
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
