from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os

api_key = os.environ["OpenAI_API_Key"] # API key is saved as a environment variable 
llm = OpenAI(openai_api_key=api_key)

def travel_idea(interest, budget):
  system_template = "You are an travel agent that specializes about {human_interest} with a budget of {human_budget} dollars."
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = "Tell me some travel ideas."
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  prompt = chat_prompt.format_prompt(human_interest=interest, human_budget=budget).to_messages()

  input_text = "\n".join([message.content for message in prompt])
  response = llm(input_text)
  return response

print(travel_idea(interest="vacation", budget="5.000"))
