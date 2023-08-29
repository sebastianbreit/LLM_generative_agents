# LLM_generative_agents

## Introduction
Our project aims to explore and build on the concept of "generative agents" by Park et al., 2023, which uses large language models to simulate virtual agents with stable long-term memories and goals.

There exist several different implementations, each with its own purposes, advantages, and disadvantages.
Some popular implementations can be found on langchain, for example the 
[implementation from the original paper by Park et al., 2023](https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters), [CAMEL an implementation of role-playing autonomous cooperative agents](https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/camel_role_playing) (Li et al., 2023), as well as other agent implementations, mostly focused on role-playing and storytelling.

What all those agents typically have in common is an implementation of a memory module of any sort, which enables successful long-term storage of information (Feng et al., 2023) beyond what is typically  inherent in chat models themselves.
In addition to a long-term memory storage, most of these agents implement a reflection or summary module of some kind (Feng et al., 2023) that is able to access the information from the memory module and reduce a large amount of conversation history down to the most important aspects.

Finally, to actually find information that is relevant to an individual request, there must be a system to prioritize statements.
Existing systems  (Li et al., 2023; Park et al., 2023; Feng et al., 2023) implemented various combinations of time-based recency weighting, similarity-based relevance weighting, and an overall importance weighting independent of the request.

Similarly to these existing implementations, we aim to explore the aspects of long-term memory and similarity-based relevance in order to prototype and test how such implementations could be used to enhance Non-Player-Characters (NPC) in video games.

In addition to agents maintaining a stable long-term memory, another important aspect of NPCs in games is their interaction abilities with the player.
In many classical video games, NPCs are subjected to gameplay restrictions, for example by predefining a small set of possible interactions with the player. This is mostly due to the fact that for each interaction, developers need to implement some logic to react to player input and develop suitable animations.
Since LLMs are able to react to any given player input, this theoretically might eliminate or greatly reduce work on gameplay logic, enabling more flexible and realistic interactions.
However, for a graphics-based game, many of those interactions still need to be animated, which is practically impossible.
Therefore, unless gameplay is restricted to a text-based interface only, there still exists a need to create restraints for the interactions of a language model based NPC with the player.

From research on language model security, it has been found that generally prefix injection and refusal suppression (Wei et al., 2023) can be used to instruct a model to follow a particular way of answering, bypassing even fine-tuned security measures.

Both mechanisms work very similarly. 
Prefix injection describes the process of instructing the model to start their answer in a particular way, for instance, "Sure, here is..." in response to a request that it typically would not fulfill.
Given that language models work with sentence completion mechanisms, this type of prompting has been shown to be generally very successful for many models. (Wei et al., 2023)
Similarly, refusal suppression works by specifically instructing the model on what not to do, requesting the model to limit its responses in a specific way, which models that are trained to be helpful generally try to accommodate.

As we can see, these mechanisms seem to be well suited for the task of limiting a language model's possible ways of interacting with the player. 
Therefore, we will implement a module that employs these methods in order to guide player interactions.


## Modules
### Dialogue
### Transaction 

For this reason, we implemented a prototype of such a transaction module that can restrict a language model's interaction possibilities and guide its responses.
Given the limited scope of this work, we implemented the single use case of a merchant transaction, where the player should only be able to buy items, which the merchant actually has in stock.

We used the memory class to simulate a basic inventory function, where the memory contains only singular words of items that should be available for purchase.
When evaluating whether or not the player should be able to purchase a specific item from the NPC, we evaluate the similarity score of the player's query against the items in memory and compare them against some arbitrary threshold.
Depending on whether the similarity score is larger or smaller than that threshold, we use prefix injection to guide the language model's response to either approve or reject that request.
The notebook **transaction_showcase** gives 2 positive and 2 negative examples of how such a guided interaction might look like.

In order to determine a suitable threshold that is actually indicative of the similarity between player query and NPC memory, we simulated data using different query and memory designs and evaluated their impact on the similarity rating.
We varied queries by changing the action verbs and the concrete items that were requested. We varied the memory content by changing the item descriptions (abstract item / concrete item) and action verbs.

Below you can see the outcome of that evaluation. The yellow color shows the cases where a transaction should occur because the queried item and action are encoded in memory, while the blue color depicts the instances where this is not the case.
![Memory impact](data/transaction_boxplots_memory.png?raw=true "Evaluating impact of memory definitions")
This first figure shows the impact of the definition of the items in memory on the similarity score. 
The left panel shows the impact of item definitions, where we see that concrete item definitions such as "axe", or "sword" result in much higher similarity scores when the query contains that word, compared to a more abstract definition in memory such as "medieval metal weapon". 
Further, the difference between cases where a transaction should happen and when it should not increase a lot.
The right panel shows that neither different action verbs (buy, sell/forge/trade) nor the omission of a verb in memory (empty) has a different impact on the similarity score. 


![Query impact](data/transaction_boxplots_query_simplified.png?raw=true "Evaluating impact of query definitions")
We also evaluated the impact of the query definition, shown in this second figure. 
We found a difference between different types of items (medieval metal weapon / other medieval weapon / armor / random items), where the first type, which should lead to a transaction, received a higher similarity than the others. In contrast, the other types all received similar scores.
The use of different action verbs (buy/forge/sell/trade) resulted in no difference. 

Based on these results, we defined our memory to simply include the available items without any actions associated with them.
Using this implementation in the **transaction_showcase** notebook, we see that the transaction module can correctly determine, whether or not it should approve or reject a request and prepend the prompt to the language model with the appropriate prefix.
While adding the correct prefix was generally successful, the language model followed that instruction mainly only for the first sentence that we restricted. Any sentence afterward seemed already less restricted, resulting in inconsistent and contradictory statements.
A possible explanation for this behavior could be that we used a basic language model that was not finetuned with reinforcement learning to be helpful and follow instructions. Due to limited local hardware, we were not able to compare this behavior against a larger and better fine-tuned model.

### Summary

## Discussion