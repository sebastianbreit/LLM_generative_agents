from sentence_embedding import Memory
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    """
        sentence_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder='all-MiniLM-L6-v2'
        )
        """

    sentence_model = SentenceTransformer(
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        cache_folder='paraphrase-MiniLM-L6-v2'
    )

    transaction_memory = Memory(
        ['I want to buy the long sword', 'Give me the long sword'],
        [1, 1]
    )

    queries = [
        'How are you?',
        'The weather is fine today',
        'I would like to purchase the long sword',
        'I would like to purchase the sword',
        'I want to buy the helmet',
        'I want to buy the short sword',
        'I want to buy the sword',
        'Give me the helmet',
        'How much for the long sword?'
    ]

    threshold = 0.75    # Arbitrary

    for query in queries:
        print(query)
        similarity_scores = transaction_memory.get_similarity_scores(query, sentence_model, normalize=False)
        print(f'Scores: {similarity_scores}')
        print(f'Average: {np.mean(similarity_scores)}')
        similarity_scores = transaction_memory.get_similarity_scores(query, sentence_model)
        print(f'Scores: {similarity_scores}')
        print(f'Average: {np.mean(similarity_scores)}')
        if threshold <= np.mean(similarity_scores):
            print('> Interaction!')
        print('\n')





























main()