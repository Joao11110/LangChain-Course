from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime
from langchain.llms import OpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain_openai import ChatOpenAI
import os

api_key = os.environ["OpenAI_API_Key"]
chat = ChatOpenAI(openai_api_key=api_key)

class HistoryQuiz():

    def createHistoryQuestion(self, topic):
        system_template = "You are a quiz maker, create a question about: On what date the {topic} happened. Only return the quiz question."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_message = "{question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        quiz = "Give me a quiz questiion where the correct anwser is a especific date"
        request = chat_prompt.format_prompt(topic=topic, question=quiz).to_messages()

        result = chat.invoke(request)
        print(result.content)
        return result.content

    def getAIAnswer(self, question):
        output_parser = DatetimeOutputParser()
        system_template = "You going to anwser quiz questions with just a date in datetime format."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = '''Anwser the question:
        {question}
        {format_intructions}'''
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        format_intructions = output_parser.get_format_instructions()

        request = chat_prompt.format_prompt(question=question, format_intructions=format_intructions).to_messages()
        result = chat.invoke(request)
        correct_datetime = datetime.strptime(result.content, "%Y-%m-%dT%H:%M:%S.%fZ")
        return correct_datetime

    def getUserAnswer(self, question):
        year = int(input("Type the year (YYYY): "))
        month = int(input("Type the month (MM): "))
        day = int(input("Type the day (DD): "))
        user_datatime = datetime(year, month, day)
        return user_datatime

    def checkUserAnswer(self, user_answer, ai_answer):
        difference = user_answer - ai_answer
        formatted_difference = str(difference)
        print(f"The difference between the anwser and you guess is: {formatted_difference}")

quiz_bot = HistoryQuiz()
question = quiz_bot.createHistoryQuestion(topic="History")
ai_answer = quiz_bot.getAIAnswer(question)
user_answer = quiz_bot.getUserAnswer(question)
quiz_bot.checkUserAnswer(user_answer, ai_answer)
