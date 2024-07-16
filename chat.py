
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
#from form import predict_insurance
#age,sex,bmi,children,smoker,region



def base(query:str):
    api_key="AIzaSyDebvFeAS_ECJlbKImDBl4mqQRPgg6zEjQ"



    #LLM loading. Here we use gemini
    llm= ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key)



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

def predict():
    pass

def explain():
    pass

def main():
    #Agent Area
    #google_api_key='AIzaSyDebvFeAS_ECJlbKImDBl4mqQRPgg6zEjQ'
    analyzer=genai.GenerativeModel("gemini-pro")

    System_prompt = """
    Answer the following questions and obey the following commands as best you can.

    You have access to the following tools:

    Plot: useful for when you need to see the relationship between two numeric element. You should ask targeted questions.
    Hisplot: Useful for when you need to a distribution ,the quantity of each categories whatever numeric data or categorial data. Use python code throughout hist for matplotlib.
    Response To Human:When you just need to respond to the human you are talking to.You use your database to answer and if the response requires other descriptions we can use an histpolt or a simple plot function to powered your analysis.


    You will receive a message from the human, then you should by Response To Human Tools and after your will choose if it necessary the appropriate plot tool among Plot and Hisplot to powered your analysis.

    Option : You use a tool to answer the question.
    For this, you should use the following format:
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Plot,Hisplot,Response To Human]
    Action Input: "the input to the action, to be sent to the tool"

    After this,he  will respond with an observation, and you will continue.

    Begin!
    """
    def system_text(system:str,message:str):
        return system+" \n" + f"human request:'{message}' " +"\n You must answer by using previous order and example"

    def Stream_agent(prompt):


    #message=[glm.Content(role=m["role"],parts=[glm.Part(text=m["content"])]) for m in message]

        def system_text(system:str,prompt:str):
            return system+" \n" + f"human request:'{prompt}' " +"\n You must answer by using previous order and example"
        

        def extract_action_and_input(text):
            action_pattern=r"Action: (.+?)\n"
            input_pattern=r"Action Input: \"(.+?)\""
            action=re.findall(action_pattern,text)
            action_input=re.findall(input_pattern,text)
            return action,action_input
        
            message="What can you say me?"


            response_text=analyzer.generate_content(system_text(system=System_prompt,message=message)).text
            print(response_text)

            #
            # time.sleep(20)
            tool=None
            action,action_input=extract_action_and_input(response_text)
            if predict[-1]=="Predict":
                tool=plot(action[-1])
            elif action[-1]=="Explain":
                tool=explain(action[-1])
            elif action[-1]=="Response To Human":
                tool=base(action[-1])
            else:
                print(action_input[-1])

            observation=tool
            print("Observation:",observation)



  

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
    main()
    response = base(user_input)
    st.session_state.messages.append(f"InSu: {response}")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    st.write(message)
