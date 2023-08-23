import numpy as np


class Memory:
    def __init__(self, key_sentences, key_sentence_ratings=None):
        self.key_sentences = key_sentences
        if(key_sentence_ratings!=None):
            self.key_ratings = key_sentence_ratings
        else:
            self.key_ratings=np.ones(len(key_sentences))

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
        '''
        get_priority returns the location of the highest prority sentences, ordered from least to highest prio
        '''
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
        prompt = ','.join(priority_sentences)

        return prompt
