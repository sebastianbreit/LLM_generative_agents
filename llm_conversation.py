from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os


"""
conda create -n game python=3.10
conda activate game
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge transformers
conda install -c conda-forge sentencepiece
conda install -c conda-forge accelerate 
conda install -c conda-forge bitsandbytes 
"""

def main():

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_3b_v2')
    model = LlamaForCausalLM.from_pretrained(
        'openlm-research/open_llama_3b_v2',
        load_in_8bit=True,
        device_map='auto',
        cache_dir='open_llama_3b_v2'
    )

    """
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = LlamaForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        load_in_8bit=True,
        device_map='auto',
        cache_dir='Llama-2-7b-hf'
    )
    """

    npc_name = 'Balgruuf the Greater'
    player_name = 'Player'

    initial_prompt = f"{npc_name} is the jarl of Whiterun. {npc_name} opposes the Imperial Empire and supports the Stormcloaks. {npc_name} is concerned about the dragons and wants to protect Whiterun from the dragons.\n"

    conversation = initial_prompt

    while True:
        user_input = input()
        # print("\n Player: " + user_input)

        conversation = conversation + f"\n{player_name}: " + user_input + f"\n{npc_name}: "
        conv_length = len(conversation)

        input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to('cuda')
        generation_output = model.generate(
            input_ids=input_ids, max_new_tokens=64, temperature=0.1
        )

        output = tokenizer.decode(generation_output[0])
        split_string = output[conv_length:].split(f'{player_name}:', 1)
        response = split_string[0]
        # print("\n {npc_name}: " + response)

        conversation += response

        print('#################')
        print(conversation)


main()
