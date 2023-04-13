######################################################################
#### Database that maintains a comprehensive record of an agent's experience.
#### From the memory stream, records are retrieved as relevant to plan the agent's actions and react appropriately
#### to the environment, and records are recursively synthetized into higher- and higher-level observations that 
#### guide behavior. 


#### The memory stream maintains a comprehensive record of an agent's experience. It is a list of memory objects, where 
#### each object contains a natural language description, a creation timestamp and a most recent timestamp, as well as 
#### agent name and the name of the user that interacted with the agent. 
#### The most basic element of the memory stream is an observation, an event directly perceived by the agent. 

#### The architecture implements a retrieval function that takes the agent's current situation as input and returns a subset
#### of the memory stream to pass on to the language model. It depends on what it is important that the agent consider when 
#### deciding how to act. Then, importance considers three main components:
#### 1. Recency: Assigns a higher score to memory objects that were recently accessed, so that events from a moment ago 
#### are likely to remain in the agent's attentional sphere. Recency as an exponential decay function over the number of hours
#### of consult (sandbox game hours) since the memory was last retrieved. The decay factor is 0.99.
####
#### 2. Importance: Distinguishes mundane from core memories, by asigning a higher score to those memory objects that the 
#### agent believes to be important. Directly asking the language model to output an integer score.
#### 
#### 3. Relevance: Assigns a higher score to memory objects that are related to the current situation. It depends on the answer to 
#### 'Relevant to what?', so we condition relevance on a query memory. Use the language model to generate an embedding vector of the 
#### text description of each memory. Then, we calculate relevance as the cosine similarity between the memory's embedding vector
#### and the query memory's embedding vector. 

#### To calculate the final retrieval function scores all memories as a weighted combination of the three elements:
#### score = \alpha_{recency} * recency + \alpha_{importance} * importance + \alpha_{relevance} * relevance
#### In the original implementation, all alphas are set to 1
#####################################################################

from langchain import OpenAI, PromptTemplate, LLMChain
from src.keys.openai_keys import openai_key
import asyncio



# TODO: Modify the examples and the word poignant
IMPORTANCE_TEMPLATE = """ On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance),
rate the likely poignancy of the following piece of memory.

Memory: {memory}
Rating: <fill in>"""


llm = OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=300, openai_api_key=openai_key)

importance_prompt = PromptTemplate(
    input_variables=["memory"],
    template=IMPORTANCE_TEMPLATE,
)

chain = LLMChain(llm=llm, prompt=importance_prompt)

def eval_importance(memory):
    resp = chain.run({"memory": memory})
    return resp

async def async_eval_importance(chain, memory):
    resp = await chain.arun({"memory": memory})
    return resp


def eval_concurrently(chain, memories, tema, textos, summary=False):
    if summary:
        tasks = [
            async_eval_importance(chain, memory)
            for memory in memories
        ]
    else:
        tasks = [
            async_eval_importance(chain, memory)
            for memory in memories
        ]
    return tasks


