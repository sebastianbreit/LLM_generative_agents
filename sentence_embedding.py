from sentence_transformers import SentenceTransformer
import numpy as np

import os
MODEL_CACHE_PATH = os.getenv("HF_MODEL_CACHE_PATH")
if MODEL_CACHE_PATH is None:
    MODEL_CACHE_PATH=''


class Memory:
    def __init__(self, key_sentences, key_sentence_ratings):
        self.key_sentences = key_sentences
        self.key_ratings = key_sentence_ratings

    def __len__(self):
        return len(self.key_sentences)

    def add_sentence(self, sentence, rating):
        self.key_sentences.append(sentence)
        self.key_ratings.append(rating)

    def get_similarity_scores(self, query, sentence_model, normalize=True):
        # Embed keys
        key_embeddings = sentence_model.encode(self.key_sentences)
        # print(key_embeddings)

        # Embed query
        query_embedding = sentence_model.encode(query)
        # print(query_embedding)

        # Get similarity scores
        sentence_similarity_scores = key_embeddings @ query_embedding

        if normalize:
            # Get vector norms
            key_embedding_norms = np.linalg.norm(key_embeddings, axis=1)
            query_embedding_norm = np.linalg.norm(query_embedding)

            # Normalize sentence similarity scores
            sentence_similarity_scores = sentence_similarity_scores / (key_embedding_norms * query_embedding_norm)

        return sentence_similarity_scores

    def get_priority(self, query, sentence_model, weighted, verbose=False):

        sentence_similarity_scores = self.get_similarity_scores(query, sentence_model)

        if weighted:
            # Weight similarity score according to importance ratings
            weighted_sentence_similarity_scores = np.multiply(sentence_similarity_scores, np.asarray(self.key_ratings)/10)
            # Sort similarity score indices
            priority = np.argsort(weighted_sentence_similarity_scores)  # Highest similarity last
            if verbose:
                print(f'Weighted Sentence Similariry Scores: {weighted_sentence_similarity_scores}')
                print(f'Weighted Priority: {priority}')

        else:
            # Sort similarity score indices
            priority = np.argsort(sentence_similarity_scores)  # Highest similarity last
            if verbose:
                print(f'Unweighted Sentence Similariry Scores: {sentence_similarity_scores}')
                print(f'Unweighted Priority: {priority}')

        return priority  # Return sorted priority indices

    def generate_prompt(self, query, sentence_model, num_sentences, weighted=True):
        # Get priority indices
        priority = self.get_priority(query, sentence_model, weighted=weighted)

        # Get priority sentences
        priority_sentences = [self.key_sentences[i] for i in priority][-num_sentences:]

        # Concatenate sentences to one prompt
        prompt = ''.join(priority_sentences)

        return prompt


# Same function as in CharacterPrompts
def get_priority(query, keys, key_ratings, sentence_model, weighted, verbose=True):

    # Embed keys
    key_embeddings = sentence_model.encode(keys)
    # print(key_embeddings)

    # Embed query
    query_embedding = sentence_model.encode(query)
    # print(query_embedding)

    # Get similarity scores
    sentence_similarity_scores = key_embeddings @ query_embedding

    # Get vector norms
    key_embedding_norms = np.linalg.norm(key_embeddings, axis=1)
    query_embedding_norm = np.linalg.norm(query_embedding)

    # Normalize sentence similarity scores
    sentence_similarity_scores = sentence_similarity_scores / (key_embedding_norms * query_embedding_norm)

    if weighted:
        # Weight similarity score according to importance ratings
        weighted_sentence_similarity_scores = np.multiply(sentence_similarity_scores, np.asarray(key_ratings) / 10)
        # Sort similarity score indices
        priority = np.argsort(weighted_sentence_similarity_scores)  # Highest similarity last
        if verbose:
            print(f'Weighted Sentence Similariry Scores: {weighted_sentence_similarity_scores}')
            print(f'Weighted Priority: {priority}')

    else:
        # Sort similarity score indices
        priority = np.argsort(sentence_similarity_scores)  # Highest similarity last
        if verbose:
            print(f'Unweighted Sentence Similariry Scores: {sentence_similarity_scores}')
            print(f'Unweighted Priority: {priority}')

    return priority     # Return sorted priority indices


def main():

    """
    sentence_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        cache_folder='all-MiniLM-L6-v2'
    )
    """

    sentence_model = SentenceTransformer(
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        cache_folder=MODEL_CACHE_PATH+'paraphrase-MiniLM-L6-v2'
    )

    # Key sentences that can be used for prompting
    key_sentences = [
        'The cat is black.',
        'The dog is white.',
        'The cat does not like the dog.',
        'The dog has a green ball.',
        'The cat has a mouse.',
        'The cat has a brother.',
        'The dog has two sisters.',
        'The cats brother is a horse.',
        'The dog likes to eat carrots.',
        'The cat likes to climb trees.'
    ]
    print(key_sentences)

    # Importance rating for key sentences
    key_sentence_ratings = [
        3, 3, 8, 5, 5, 7, 7, 5, 7, 7
    ]

    # Number of sentences used for prompting
    num_sentences = 3

    # Query sentences used to showcase the priority rating
    query_sentence_examples = [
        'Which color is the cat?',
        'Which color is the dog?',
        'Where does the cat come from?',
        'Where does the dog live?',
        'Do you know a horse?',
        'Does this belong to you?',
        'Do you have siblings?',
        'What would you like to do?',
        'What do you want to do?',
        'Do you have enemies?'
    ]

    for query_sentence in query_sentence_examples:

        print(f'\nQuery: {query_sentence}')

        priority = get_priority(query_sentence, key_sentences, key_sentence_ratings,
                                sentence_model, weighted=False, verbose=True)

        priority_sentences = [key_sentences[i] for i in priority][-num_sentences:]
        print(f'Unweighted Priority: {priority_sentences}')

        prompt = ' '.join(priority_sentences)
        print(f'Unweighted Prompt: {prompt}')

        priority = get_priority(query_sentence, key_sentences, key_sentence_ratings,
                                sentence_model, weighted=True, verbose=True)

        weighted_priority_sentences = [key_sentences[i] for i in priority][-num_sentences:]
        print(f'Weighted Priority: {weighted_priority_sentences}')
        prompt = ' '.join(weighted_priority_sentences)
        print(f'Weighted Prompt: {prompt}')


main()


