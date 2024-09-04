from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
import os

api_key = os.environ["OpenAI_API_Key"]
chat = ChatOpenAI(openai_api_key=api_key)

def anwser_question_about(person_name, question):
    loader = WikipediaLoader(query=person_name, load_max_docs=1)
    context = loader.load()[0].page_content

    template = "Anwser this question:\n{question}\nHere is some context about it:\n{document}"
    human_prompt = HumanMessagePromptTemplate.from_template(template)
  
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    result = chat.invoke(chat_prompt.format_prompt(question=question, document=context))
    print(result.content)

anwser_question_about("Albert Einsten","When was he born?")
