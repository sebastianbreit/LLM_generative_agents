from memory import Memory
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

MODEL_CACHE_PATH = os.getenv("HF_MODEL_CACHE_PATH")
if MODEL_CACHE_PATH is None:
    MODEL_CACHE_PATH=''

query_items=[
        'axe',
        'sword',
        'mace',
        'hammer',
        'dagger',

        'bow',
        'crossbow',
        'club',
        'shield',
        'slingshot',

        'helmet',
        'chestplate',
        'gauntlets',
        'bracers',
        'leg armor',

        'football',
        'air',
        'doll',
        'sandwitch',
        'computer'
    ]
item_queries=[]
buy_queries_easy=[]
buy_queries_hard=[]
for i_item,item in enumerate(query_items):
    item_queries.append([
        f'I want to trade the {item}.',
        f'I want to sell the {item}.',
        f'Can you forge the {item} for me?',
    ])

    buy_queries_easy.append([
        f'I want to buy the {item}.',
        f'I want to purchase the {item}.',
        f'I want to acquire the {item}.',
    ])
    buy_queries_hard.append([
        f'How much for the {item}?',
        f'Give me the {item}.',
        f'I want the {item}.',
    ])

random_queries = [
    'How are you?',
    'What did you eat today?',
    'I am the Dragonborn.',
]

query_list_list= [item_queries,buy_queries_easy,buy_queries_hard,random_queries]

abstract_items_memory=[
    'medieval metal weapon',
    'forged medieval weapon',
    'medieval melee weapon'
]
concrete_items_memory=[
    'axe',
    'sword',
    'mace',
    'hammer',
    'dagger'
]
combined_items_memory=abstract_items_memory+concrete_items_memory
memory_list_items=[abstract_items_memory,concrete_items_memory,combined_items_memory]

buy_synonyms_memory=[
    'buy','purchase','acquire'
]
other_action_memory=[
    'sell','forge','trade'
]
negation_memory=[
    'not buy','not purchase','not acquire'
]
empty_memory=['']
memory_list_actions=[buy_synonyms_memory,other_action_memory,negation_memory,empty_memory]

def generate_q_short(query):
  if(any(query in sublist for sublist in item_queries)): return 'alternatives'
  elif(any(query in sublist for sublist in buy_queries_easy)): return 'synonym_easy'
  elif(any(query in sublist for sublist in buy_queries_hard)): return 'synonym_hard'
  elif(query in random_queries): return 'random'
  else: return 'Error: query invalid / not found'

def generate_q_a(query):
    if any(buy_action in query for buy_action in ['buy','purchase','acquire']):
        return 'buy'
    elif ('sell' in query):
        return 'sell'
    elif ('forge' in query):
        return 'forge'
    elif ('trade' in query):
        return 'trade'
    else:
        return 'other'

def generate_q_i(query):
    for item in query_items:
        if (item in query):
            return item
    return 'No item'

def generate_success(query):
    for item in query_items[0:5]:
        if (item in query):
            return True
    return False

def generate_m_a_short(m_action):
  if((m_action == buy_synonyms_memory)): return 'buy'
  elif((m_action ==  other_action_memory)): return 'sell/forge/trade'
  elif((m_action ==  negation_memory)): return 'not buy'
  elif((m_action ==  empty_memory)): return 'empty'
  else: return 'Error'

def generate_m_i_short(m_item):
  if ((m_item ==  abstract_items_memory)):
      return 'abstract'
  elif ((m_item ==  concrete_items_memory)):
      return 'concrete'
  elif ((m_item ==  combined_items_memory)):
      return 'combined'
  else:
      return 'other / none'




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


    query_df = pd.DataFrame(columns=["q_short","q_a","q_i", "query", "m_a","m_i", "memory", "score_list", "avg_score", "max_score","exp_success"])
    memory_df=pd.DataFrame(columns=['m_a','m_i','memory_list_complete'])

    for action_list in memory_list_actions:
        for item_list in memory_list_items:
            m_a = generate_m_a_short(action_list)
            m_i=generate_m_i_short(item_list)
            memory_list_complete = []

            for action in action_list:
                for inner_item in item_list:
                    memory_list_complete.append(action+" "+inner_item)

            memory_df.loc[len(memory_df)] = [m_a,m_i,memory_list_complete]

    for index, memory in memory_df.iterrows():
        temp_memory = Memory(memory.memory_list_complete)
        for query_list in (query_list_list):
            for query_test in query_list:
                print(query_test)

                if type(query_test) is list:
                    for query in query_test:
                        append_query_to_df(query, query_df, sentence_model, transaction_memory=temp_memory,m_a=memory.m_a,m_i=memory.m_i)
                else:
                    query=query_test
                    append_query_to_df(query, query_df, sentence_model, transaction_memory=temp_memory,m_a=memory.m_a,m_i=memory.m_i)


    query_df.to_csv(path_or_buf='data/transaction_test.csv')

            # print(f'Normalized Scores: {similarity_scores}')
            # print(f'Normalized Average: {np.mean(similarity_scores)}')
            # print(f'Normalized Max: {np.max(similarity_scores)}')

            # prompt = transaction_memory.generate_prompt(query, sentence_model, num_sentences=3, weighted=True)
            # print(f'Prompts: {prompt}')

            # threshold = 0.5  # Arbitrary
            # if threshold <= np.max(similarity_scores):
            #     print('> Interaction!')
            # print('\n')


def append_query_to_df(query, query_df, sentence_model, transaction_memory,m_a=None, m_i=None):
    similarity_scores = transaction_memory.get_similarity_scores(query, sentence_model)
    q_short = generate_q_short(query)
    q_a=generate_q_a(query)
    q_i=generate_q_i(query)
    exp_success=generate_success(query)
    new_entry_list = [
        q_short,
        q_a,
        q_i,
        query,
        m_a,
        m_i,
        transaction_memory.key_sentences,
        similarity_scores,
        np.mean(similarity_scores),
        np.max(similarity_scores),
        exp_success
    ]
    # pd.concat([query_df, new_record], ignore_index=True)
    query_df.loc[len(query_df)] = new_entry_list


main()