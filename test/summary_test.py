from transformers import pipeline
from util.summary_generator import SummaryGenerator
from util.memory import Memory
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto",
                                                 torch_dtype=torch.bfloat16, cache_dir='dolly-v2-3b')

    summary_pipeline = pipeline('summarization', model='kabita-choudhary/finetuned-bart-for-conversation-summary')

    summary_generator = SummaryGenerator(summary_pipeline)

    sentence_model = SentenceTransformer(
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        cache_folder='paraphrase-MiniLM-L6-v2'
    )

    dialogue = """Laurie: So, what are your plans for this weekend?\n
    Christie: I don’t know. Do you want to get together or something?\n
    Sarah: How about going to see a movie? Cinemax 26 on Carson Boulevard is showing Enchanted. Laurie: That sounds like a good idea. Maybe we should go out to eat beforehand.\n
    Sarah: It is fine with me. Where do you want to meet?\n
    Christie: Let’s meet at Summer Pizza House. I have not gone there for a long time.\n
    Laurie: Good idea again. I heard they just came up with a new pizza. It should be good because Summer Pizza House always has the best pizza in town.\n
    Sarah: When should we meet?\n
    Christie: Well, the movie is shown at 2:00PM, 4:00PM, 6:00PM and 8:00PM.\n
    Laurie: Why don’t we go to the 2:00PM show? We can meet at Summer Pizza House at noon. That will give us plenty of time to enjoy our pizza.\n
    Sarah: My cousin Karen is in town. Can I bring her along? I hate to leave her home alone.\n
    Christie: Karen is in town? Yes, bring her along. Laurie, you remember Karen? We met her at Sara’s high school graduation party two years ago.\n
    Laurie: I do not quite remember her. What does she look like?\n
    Sarah: She has blond hair, she is kind of slender, and she is about your height.\n
    Laurie: She wears eyeglasses, right?\n
    Sarah: Yes, and she was playing the piano off and on during the party.\n
    Laurie: I remember her now. Yes, do bring her along Sara. She is such a nice person, and funny too.\n
    Sarah: She will be happy to meet both of you again.\n
    Christie: What is she doing these days?\n
    Sarah: She graduated last June, and she will start her teaching career next week when the new school term begins.\n
    Laurie: What grade is she going to teach?\n
    Sarah: She will teach kindergarten. She loves working with kids, and she always has such a good rapport with them\n
    Christie: Kindergarten? She must be a very patient person. I always think kindergarten is the most difficult class to teach. Most of the kids have never been to school, and they have e never been away from mommy for long.\n
    Sarah:  I think Karen will do fine. She knows how to handle young children\n
    Laurie: I think the first few weeks will be tough. However, once the routine is set, it should not be too difficult to teach kindergarten.\n
    Christie: You are right. The kids might even look forward to going to school since they have so many friends to play with.\n
    Sarah: There are so many new things for them to do at school too. They do a lot of crafts in kindergarten. I am always amazed by the things kindergarten teachers do.\n
    Laurie: Yes, I have seen my niece come home with so many neat stuff.\n
    Christie: Maybe we can ask Karen to show us some of the things that we can do for this Halloween.\n
    Laurie: Maybe we can stop by the craft store after the movie. What do you think, Sara?\n
    Sarah: I will talk to her. I think she will like that. It will help her with school projects when Halloween comes.\n
    Christie: Michael’s is a good store for crafts. It always carries a variety of things, and you can find almost anything there.\n
    Laurie: There is a Michaels store not far away from Cinemax 26. I believe it is just around the corner, on Pioneer Avenue. We can even walk over there.\n
    Sarah: So, we plan to meet for pizza at noon, go to the movies at two, and shop at Michael’s afterward. Right?\n
    Laurie and Christie: Yes."""

    print(dialogue)

    summary = summary_pipeline(dialogue, max_length=128, min_length=30, do_sample=False)
    print(f"Summary: {summary[0]['summary_text']}")

    summary = summary_pipeline(dialogue, max_length=128, min_length=30)
    print(f"Summary: {summary[0]['summary_text']}")

    sentence_list = summary_generator.generate_memories(summary[0]['summary_text'])
    print(f'Prompt list: {sentence_list}')

    sentence_rating_list = [1 for s in sentence_list]

    memory = Memory(list(), list())
    memory.add_sentence(sentence_list, sentence_rating_list)

    print(f'Sentences: {memory.key_sentences}')

    user_input = 'Hey, how was Karens first day at work?'

    initial_prompt = memory.generate_prompt(user_input, sentence_model, num_sentences=2)

    conversation = f"\nLaurie:\n" + user_input + f"\nSarah: "

    input_text = initial_prompt + "\n" + conversation
    input_length = len(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=64, temperature=0.2
    )

    output = tokenizer.decode(generation_output[0])
    split_string = output[input_length:].split('\nLaurie:', 1)

    response = split_string[0]

    print("\nSarah: " + response)


main()

