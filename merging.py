import pandas as pd

malicious = 'Datasets/malicious_phish.csv'
phishing = 'Datasets/phishing_sites.csv'

malicious_df = pd.read_csv(malicious)
phishing_df = pd.read_csv(phishing)

malicious_df['Label'] = malicious_df['type'].map({
    'phishing': 'bad',
    'defacement': 'bad',
    'malware': 'bad',
    'benign': 'good'
})

malicious_df.drop(columns=['type'], inplace=True)

combined_df = pd.concat([malicious_df, phishing_df])
combined_df.to_csv('combined.csv')