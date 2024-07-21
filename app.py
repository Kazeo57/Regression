import streamlit as st


def home_():
    #[theme]
    st.title("Insurance Viewer")
    st.write("*This app is a way you can use to predict insurance charge based on ***Age,Sex,bmi,Region ,Children,Smoker.****")

    st.write("You can upload csv file or excel,complete a form and even chat with a bot powered by llm to give you an appropriate prediction.")
    st.image("images/homeview.jpeg")



#home=st.Page(home_,title="Home")
#form=st.Page(formu,title="Form")
#drag_file=st.Page("drag.py",title="Drag File")
#chat=st.Page("chat.py",title="Chat")


pages=st.navigation([st.Page(home_,title="Home"),
        st.Page("form.py",title="Form"),
        st.Page("drag.py",title="Drag File"),
        st.Page("chat.py",title="Chat"),
        st.Page("doc.py",title="Overview")
        ])
    
pages.run()