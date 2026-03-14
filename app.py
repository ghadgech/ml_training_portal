import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import openai

st.set_page_config(page_title="ML Training Portal")

# --------------------------
# OPENAI API
# --------------------------


openai.api_key = st.secrets["OPENAI_API_KEY"]


# --------------------------
# LOGIN FUNCTIONS
# --------------------------

def login_user(username,password):

    try:
        users = pd.read_csv("users.csv")
    except:
        return None

    user = users[
        (users["username"]==username) &
        (users["password"]==password)
    ]

    if not user.empty:
        return user.iloc[0]

    return None


# --------------------------
# SAVE RESULTS
# --------------------------

def save_result(username,name,batch,score,total):

    result = pd.DataFrame({

    "username":[username],
    "name":[name],
    "batch":[batch],
    "score":[score],
    "total":[total]

    })

    try:

        old = pd.read_csv("results.csv")

        new = pd.concat([old,result])

    except:

        new = result

    new.to_csv("results.csv",index=False)

    return new


# --------------------------
# COURSE MODULES
# --------------------------

modules = {

1:{
"title":"Python for Machine Learning",
"content":"Python libraries used in ML include NumPy, Pandas and scikit-learn."
},

2:{
"title":"Data Preprocessing",
"content":"Preprocessing includes scaling, encoding and handling missing data."
},

3:{
"title":"Regression",
"content":"Regression models predict numeric values such as price or sales."
}

}

# --------------------------
# QUESTION BANK
# --------------------------

question_bank = [

{
"question":"Machine learning is mainly used for:",
"options":[
"Making predictions from data",
"Printing documents",
"Typing emails",
"Creating folders"
],
"answer":"Making predictions from data"
},

{
"question":"Which algorithm predicts numeric values?",
"options":[
"Linear Regression",
"K-Means",
"PCA",
"Naive Bayes"
],
"answer":"Linear Regression"
},

{
"question":"Which algorithm predicts categories?",
"options":[
"Logistic Regression",
"Linear Regression",
"PCA",
"K-Means"
],
"answer":"Logistic Regression"
}

]

def generate_quiz():

    return random.sample(question_bank,3)

# --------------------------
# SESSION
# --------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# --------------------------
# LOGIN PAGE
# --------------------------

if not st.session_state.logged_in:

    st.title("Machine Learning Training Portal")

    username = st.text_input("Username")
    password = st.text_input("Password",type="password")

    if st.button("Login"):

        user = login_user(username,password)

        if user is not None:

            st.session_state.logged_in = True
            st.session_state.username = user["username"]
            st.session_state.name = user["name"]
            st.session_state.batch = user["batch"]

            st.success("Login successful")

        else:

            st.error("Invalid credentials")

# --------------------------
# MAIN PORTAL
# --------------------------

if st.session_state.logged_in:

    st.sidebar.write(f"Student: {st.session_state.name}")
    st.sidebar.write(f"Batch: {st.session_state.batch}")

    page = st.sidebar.selectbox(

        "Menu",

        [
        "Learn Modules",
        "Take Quiz",
        "Instructor Dashboard",
        "AI Tutor"
        ]

    )

# --------------------------
# MODULES
# --------------------------

    if page=="Learn Modules":

        module = st.selectbox("Select Module",list(modules.keys()))

        st.header(modules[module]["title"])

        st.write(modules[module]["content"])

# --------------------------
# QUIZ
# --------------------------

    if page=="Take Quiz":

        st.title("Machine Learning Quiz")

        if st.button("Start Quiz"):

            st.session_state.questions = generate_quiz()

        if "questions" in st.session_state:

            questions = st.session_state.questions

            answers=[]

            for i,q in enumerate(questions):

                st.subheader(q["question"])

                ans = st.radio(

                "Choose answer",

                q["options"],

                key=i

                )

                answers.append(ans)

            if st.button("Submit Quiz"):

                score=0

                for i,q in enumerate(questions):

                    if answers[i]==q["answer"]:

                        score+=1

                st.success(f"Score: {score}/{len(questions)}")

                data=save_result(

                    st.session_state.username,
                    st.session_state.name,
                    st.session_state.batch,
                    score,
                    len(questions)

                )

# --------------------------
# DASHBOARD
# --------------------------

    if page=="Instructor Dashboard":

        st.title("Instructor Dashboard")

        try:
            df=pd.read_csv("results.csv")
        except:
            st.warning("No results yet")
            st.stop()

        st.metric("Students",df["username"].nunique())

        st.metric("Average Score",round(df["score"].mean(),2))

        st.dataframe(df.sort_values("score",ascending=False))

        fig,ax=plt.subplots()

        ax.hist(df["score"])

        st.pyplot(fig)

# --------------------------
# AI TUTOR
# --------------------------

    if page=="AI Tutor":

        st.title("AI Tutor")

        question = st.text_input("Ask a Machine Learning Question")

        if st.button("Ask AI"):

            response = openai.ChatCompletion.create(

                model="gpt-4o-mini",

                messages=[
                    {"role":"system",
                     "content":"You are a helpful machine learning tutor."},

                    {"role":"user",
                     "content":question}
                ]
            )

            answer=response["choices"][0]["message"]["content"]

            st.write(answer)