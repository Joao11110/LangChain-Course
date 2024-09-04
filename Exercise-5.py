from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os

api_key = os.environ["OpenAI_API_Key"] 
chat = ChatOpenAI(openai_api_key=api_key)

def translate_and_summarize(email):
    template1 = "Tell me ONLY the language of this email:\n{email}\nReturn the language it was written in."
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain1 = LLMChain(llm=chat, prompt=prompt1, output_key="language")

    template2 = "Translate this email from {language} to English."+email
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = LLMChain(llm=chat, prompt=prompt2, output_key="translated_email")

    template3 = "Make a summary of this email\n{translated_email}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain3 = LLMChain(llm=chat, prompt=prompt3, output_key="summary")

    sequencial_chain = SequentialChain(chains=[chain1, chain2, chain3],input_variables=['email'], output_variables=['language', 'translated_email', 'summary'], verbose=True)

    return sequencial_chain.invoke(email)


email = """Asunto: Problemas Técnicos con el Sistema

Estimado Sr. Martínez:

Espero que este correo le encuentre bien. Me pongo en contacto con usted para informar sobre algunos problemas técnicos que he estado experimentando con el sistema.

Desde hace algunos días, he notado que el sistema se ralentiza considerablemente y, en ocasiones, se congela durante el uso de ciertas aplicaciones, como el software de gestión de proyectos "ProTask". Este problema está afectando nuestra productividad y el flujo de trabajo diario en el Departamento de Finanzas.

He intentado algunas soluciones básicas, como reiniciar el sistema y actualizar el software, pero el problema persiste. Me gustaría solicitar su asistencia para solucionar este problema lo antes posible.

Por favor, hágame saber si necesita más detalles sobre el problema o si hay algún procedimiento adicional que deba seguir para ayudar a resolverlo.

Agradezco de antemano su atención a este asunto y quedo a la espera de su pronta respuesta.

Atentamente,
Laura Gómez
Coordinadora de Finanzas
laura.gomez@empresaexample.com"""

translate_and_summarize(email)
