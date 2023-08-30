from sentence_transformers import SentenceTransformer
from util.memory import Memory
import os
MODEL_CACHE_PATH = os.getenv("HF_MODEL_CACHE_PATH")
if MODEL_CACHE_PATH is None:
    MODEL_CACHE_PATH=''

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

    print(sentence_model)

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

    memory = Memory(
        key_sentences,
        key_sentence_ratings
    )

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

        priority = memory.get_priority(query_sentence, sentence_model,
                                       weighted=False, verbose=True)

        priority_sentences = [key_sentences[i] for i in priority][-num_sentences:]
        print(f'Unweighted Priority: {priority_sentences}')

        prompt = ' '.join(priority_sentences)
        print(f'Unweighted Prompt: {prompt}')

        priority = memory.get_priority(query_sentence, sentence_model,
                                       weighted=True, verbose=True)

        weighted_priority_sentences = [key_sentences[i] for i in priority][-num_sentences:]
        print(f'Weighted Priority: {weighted_priority_sentences}')
        prompt = ' '.join(weighted_priority_sentences)
        print(f'Weighted Prompt: {prompt}')


main()
