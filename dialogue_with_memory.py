from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from memory import Memory
import torch
import os
MODEL_CACHE_PATH = os.getenv("HF_MODEL_CACHE_PATH")
if MODEL_CACHE_PATH is None:
    MODEL_CACHE_PATH=''

def main():

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    """
    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_3b_v2')
    model = LlamaForCausalLM.from_pretrained(
        'openlm-research/open_llama_3b_v2',
        #load_in_8bit=True,
        device_map='auto',
        cache_dir='open_llama_3b_v2'
    )
    """

    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto",
                                                 torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE_PATH+'dolly-v2-3b')
    print(model)

    """
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = LlamaForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        load_in_8bit=True,
        device_map='auto',
        cache_dir='Llama-2-7b-hf'
    )
    """

    sentence_model = SentenceTransformer(
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        cache_folder=MODEL_CACHE_PATH+'paraphrase-MiniLM-L6-v2'
    )

    npc_name = 'Balgruuf the Greater'
    player_name = 'Player'

    memories = [
        'is the jarl of Whiterun.',
        'is a Nord'
        'resides in the great hall Dragonsreach in Whiterun.',
        'wears noble cloths.',
        'wears a crown.',
        'has a unique war axe.',
        'has a brother called Hrongar.',
        'has three children called Frothar, Dagny and Nelkir.',
        'has no wife.',
        'puts Whiteruns interests first.',
        'did not permit the Imperials to garrison soldiers in the city.',
        'takes no sides in the war.',
        'is always on the side of Whiterun',
        'does not like Urlfric and Galmar, because they attacked Whiterun.',
        'does not like the stormcloaks, because they attacked Whiterun.',
        'is concerned about the dragons, because they attacked Whiterun.',
        'worships Talos, the god of the Nords.',
        'hates the Thalmor.',
        'is friends with Irileth, because he fought with her in the war.',
    ]

    memories = [npc_name + " " + m + " " for m in memories]

    memory_ratings = [4, 1, 2, 1, 2, 3, 3, 2, 5, 2, 3, 4, 3, 3, 5, 2, 4, 4]

    memory = Memory(memories, memory_ratings)

    num_sentences = 4

    conversation = ""

    while True:
        user_input = input('Player: ')
        #print("\nPlayer: " + user_input)

        initial_prompt = memory.generate_prompt(user_input, sentence_model, num_sentences=num_sentences)
        #print(f'\nInitial Prompt: {initial_prompt}')

        conversation = conversation + f"\n{player_name}:\n" + user_input + f"\n{npc_name}: "

        input_text = initial_prompt + "\n" + conversation
        input_length = len(input_text)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        generation_output = model.generate(
            input_ids=input_ids, max_new_tokens=64, temperature=0.2
        )

        output = tokenizer.decode(generation_output[0])
        split_string = output[input_length:].split(f'\n{player_name}:', 1)
        response = split_string[0]
        #print(f"\n{npc_name}: " + response)

        conversation += response
        input_text += response

        # Print full text each step:
        print('#################')
        print(input_text)


main()