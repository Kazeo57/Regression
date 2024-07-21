
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from google.generativeai import client 
import google.generativeai as genai
import re
import json
import ast
from form import predict_insurance
#from form import predict_insurance
#age,sex,bmi,children,smoker,region



def base(query:str):
    api_key="AIzaSyDebvFeAS_ECJlbKImDBl4mqQRPgg6zEjQ"



    #LLM loading. Here we use gemini
    llm=ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key)



    #Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)




    #BDV loading
    data_store=Chroma(persist_directory="center_bdv", embedding_function=embeddings)





    #Then, we will retriever a vectorial database. If you want how to xreate a database, you'll find it in notebook
    retriever = data_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})



    #PrompTemplate
    #Here I create a template to use to query the model
    template = """Tu es un chatbot qui répond ax préccupations de l'utilisateur relative aux facteurs liés au montant de la prise en charge de l'assurance médicale.
    Ces facteurs sont entre autre le sexe,la masse corporelle(bmi),le nombre d'enfant à charge,le sexe du garant,la région de provenance et l'état de l'indivi(s'il fume ou non).
    Si tu n'as pas réponses à une question,dis que la question est un peu superflu pour toi et utilise les connaissances de gemini pour analyser,comprendre les besoin de l'utiklisateur puis lui répondre.
    context: {context}
    input: {input}
    answer:
    """


    prompt = PromptTemplate(
            template=template,
        input_variables=['input']
    )



    #Chain LLM, prompt and retriever
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)





    #Let's write a function to retrieve with llm
    
    response=retrieval_chain.invoke({"input": query})

    if response:
        return response['answer']
    else :
        "Please, ask another question"

def predict(attribute_list):
    age,sex,bmi,children,smoker,region=attribute_list
    #if "null" in query.values():
    '''
    try:
        prediction=predict_insurance(age,sex,bmi,children,smoker,region)
    except:
        prediction="Vous n'avez pas donné de valeur à l'une des caractéristiques pour la  prédiction" '''

    prediction=predict_insurance(age,sex,bmi,children,smoker,region)   
    return f"En nous basant sur vos données on peut dire que le montant qui correspond au profil décrit est: {prediction[0]}"

def explain(query):
    return "explanation"


#api_key='AIzaSyDebvFeAS_ECJlbKImDBl4mqQRPgg6zEjQ'
api_key="AIzaSyA6J3u2hL4jNsFE67yy2q75KV9BSkdLHnI"
analyzer=genai.GenerativeModel("gemini-pro")
genai.configure(api_key=api_key)
System_prompt = """
    If the human language is not English ,then translate firstly.After that answer the following questions and obey the following commands as best you can.

    You have access to the following tools:

    Predict: useful for when you need to make a prediction based on age,sex,bmi,children,smoker and region.if one of these element is null say  directly to user what is omitted else if more elements are given use only the necessary 
    Explain: Useful for when you need to explain a prediction or behaviour for charges in insurance cases on basis of your data to answer.
    Response To Human:When you need to respond to the human you are talking to.
    You will receive a message from the human, then  you should choose between one of these two options  to powered your answer.

    Option 1 : You use a tool to answer the question.
    For this, you should use the following format:
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Predict,Explain]
    Action Input: "the input to the action, to be sent to the tool.Warning: especially in predict use the format {'age':21,'sex':'male','bmi':16,'children':2,'smoker':yes/no,'region':'southeast'}"

    After this,the human  will respond with an observation, and you will continue.

    Option 2: You respond to the human.
    For this,you should use the following format:
    Action:Response To Human
    Action Input: your response to the human, summarizing what you did and what you learned"
    
    Begin!
    """
#print(analyzer.generate_content("Comment tu vas").text)
    

   
   

def Stream_agent(prompt):


    #message=[glm.Content(role=m["role"],parts=[glm.Part(text=m["content"])]) for m in message]

    def system_text(system:str,prompt:str):
        return system+" \n" + f"human request:'{prompt}' " +"\n You must answer by using previous order and example"
    

    def extract_action_and_input(text):
        action_pattern=r"Action: (.+?)\n"
        input_pattern=r"\{.*\}"
        action=re.findall(action_pattern,text)
        action_input_match=re.search(input_pattern,text)
        action_input=action_input_match.group()
        #print("match",action_input)
        #action_input_match = re.sub(r'(\w+):', r'"\1":', action_input_match)
       # action_input_match= action_input_match.replace('\'', '\"').replace(': ', ':').replace(', ', ',')
        #pattern = r'=(\s*[^,]*)'
        #action_input = re.findall(pattern, action_input_match)
        action_input=ast.literal_eval(action_input)
        return action,action_input
    
        #message="What can you say me?"

    response=analyzer.generate_content(prompt)
    response_text=analyzer.generate_content(system_text(system=System_prompt,prompt=prompt)).text
    print(response_text)

    #
    # time.sleep(20)
    tool=None
    action,action_input=extract_action_and_input(response_text)
    print("actttt",action)
    print("inputtt",list(action_input.values()))
    #try:
    if action[-1]=="Predict":
        print("action input",action_input)
        print(action_input)
        tool=predict(action_input.values())
        #print(tool)
        #print("tool",tool)
    elif action[-1]=="Explain":
        tool=explain(action[-1])

    elif action[-1]=="Response To Human":
        #print(action_input[-1])
        tool=action_input[-1]
    observation=tool
    #except:
     #observation="Pour effectuer cette prédiction il faut nécessairemnt 5 caractéristiques soient : 'sex', 'bmi', 'children', 'smoker', et 'region' ."

   
    print("Observation:",observation)
    return observation 
#print(Stream_agent("Explain me how charges is given to person in insurance case"))
#print(Stream_agent("Je voudrais que tu me prédises le montant à partir d'un IMC de 16. Il faut aussi savoir que la personne fume, a 2 enfants à charge, et son garant est une femme.Il vient de sud-Est et à 21ans"))
print(Stream_agent("Je voudrais que tu me prédise le montant à partir d'un bmi de 16,un sexe masculin,il faut ausssi savoir que la persone fume, a 2 enfants à charge et soon garant une femme.il vient du Sud-Est,il a 21 ans"))

  
"""
# Configuration de l'interface

st.title("InSu")
st.write("What do you know? I can answer on a lot of questions about Insurance charges especially factors causing it like sex,age,smoking state,bmi and regions.")
#st.image("A..jpg",width=100)
# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# Saisie utilisateur
user_input = st.text_input("You:", "")

# Si l'utilisateur a entré un message, ajoutez-le à l'historique et obtenez la réponse du chatbot
if user_input:
    st.session_state.messages.append(f"You: {user_input}")
    #main()
    response = Stream_agent(user_input)
    st.session_state.messages.append(f"InSu: {response}")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    st.write(message)
"""

#print(main("Comment tu vas?"))