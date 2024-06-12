
search_terms = ['machine learning', 'deep learning', 'artificial intelligen', 'genetic algorithm', 'support vector', 'image segmentation',
                'particle swarm', 'reinforcement learning', 'random forest', 'computer vision', 'transfer learning',
                'natural language processing', 'image classification', 'supervised learning', 'attention mechanism',
                'semantic segmentation', 'generative adversarial network',
                'facial recognition', 'sentiment analysis', 'multi-agent system', 'multiagent system', 'ensemble learning', 'extreme learning',
                'recommender system', 'image retrieval', 'decision tree', 'image fusion', 'long short-term memory',
                'evolutionary algorithm', 'ant colony optimization', 'convolutional neural network', 'artificial neural network',
                'deep neural network', 'recurrent neural network', 'bp neural network', 'graph neural network']
import re
import json
import math


def normalize_keyword_hits(num_hits, k=8, threshold=0.5):
    # normalized_score = 1 - 1 / (1 + math.exp(k * (num_hits - threshold)))
    normalized_score = 0
    if num_hits > 0:
        normalized_score = 1
    return normalized_score

# Function to count matches for a list of patterns in a given text
def count_matches(text, list):
    res = 0
    for key in list:
        if text.count(key) > 0:
            res = res + 1
    return res

index = 0


# Open the JSONL file and output file
with open('data/manual_labelled.jsonl', 'r') as f, open('data/keywords_match.csv', 'w') as out_f:
    out_f.write('sample_id,prediction_probability\n')
    for line in f:
        index = index + 1
        print(index)
        sample = json.loads(line)

        # Combine keywords, title, and abstract for matching
        text = ' '.join([sample['keywords'], sample['title'], sample['abstract']])

        # Count matches for each list of patterns
        all_counts = normalize_keyword_hits(count_matches(text, search_terms))

        out_f.write(sample['id'] + ',' + str(all_counts) + '\n')

