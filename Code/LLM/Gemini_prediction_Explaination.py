import os
import pandas as pd
import joblib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import csv
import matplotlib.pyplot as plt

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}


def read_csv_data(csv_files):
    all_data = [] 

    for csv_file in csv_files:
        csv_text = csv_file.read().decode("utf-8")
        csv_reader = csv.reader(csv_text.splitlines())

        headers = next(csv_reader) 
        for row in csv_reader:
            if row:  
                row_data = {headers[i]: row[i] for i in range(len(row))}  
                all_data.append(row_data)

    return all_data  


def get_text_chunks(data):
    text = ""
    
    for entry in data:
        formatted_entry = ' '.join(f"{key}: {value}" for key, value in entry.items())
        text += formatted_entry + "\n"  

    splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
        Based on the detected timestamp anomalies in the dataset, please perform a root cause analysis considering the following:
        Anomalies detected in timestamps related to specific activities:
        {context}

        Please provide insights on :
        1. Possible reasons for these anomalies.
        2. Impact of these anomalies on the overall process or system.
        3. Recommendations for handling or preventing similar anomalies in the future.
        Context: \n{context}\n
        Question: \n{question}\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some CSVs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response


def get_timestamps_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    if 'time:timestamp' not in df.columns:
        st.error("The required column 'time:timestamp' is not present in the CSV file.")
        return None, None

    timestamps = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', dayfirst=True)
    activities = df['concept:name']  
    return timestamps, activities


def create_features(timestamps):
    features = pd.DataFrame()
    features['timestamp'] = timestamps
    features['hour'] = timestamps.dt.hour
    features['day_of_week'] = timestamps.dt.dayofweek
    features['day_of_month'] = timestamps.dt.day
    features['month'] = timestamps.dt.month
    features['year'] = timestamps.dt.year
    features['time_diff'] = timestamps.diff().dt.total_seconds().fillna(0)
    return features


def detect_anomalies(model, features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features[['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'time_diff']])
    predictions = model.predict(scaled_features)
    anomalies = features[predictions == -1]
    return anomalies


def main():
    st.set_page_config(
        page_title="Gemini CSV Chatbot",
        page_icon="🍀"
    )

    with st.sidebar:
        st.title("Menu:")
        csv_docs = st.file_uploader("Upload your CSV Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True, type="csv")


    st.title("ProM-Ex 🍀")
    st.write("Welcome! ")
  
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a CSV file, and I will detect and explain their anomalies."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

   
    if st.sidebar.button("Submit & Process Anomalies"):
        with st.spinner("Processing..."):
            if csv_docs:
                timestamps, activities = get_timestamps_from_csv(csv_docs[0])  \
            else:
                st.error("Please upload a valid CSV file.")
                return

    
            model = joblib.load("/Project/timestamp_anomaly_activity_detection_model.pkl copy")

            features = create_features(timestamps)
            features['concept:name'] = activities 
            anomalies = detect_anomalies(model, features)

            st.success("Processing complete.")

            if not anomalies.empty:
                st.subheader("Detected Timestamp Anomalies:")
                st.write(anomalies[['timestamp', 'concept:name']])

                
                context = anomalies.to_string()
                prompt = f"The following timestamps were detected as anomalies:\n{anomalies['timestamp'].tolist()}\nPlease provide an explanation."
                chain = get_conversational_chain()

                documents = [Document(page_content=context)]
                response = chain({"input_documents": documents, "question": prompt}, return_only_outputs=True)

                st.write(response['output_text'])

                
                st.subheader("Anomaly Detection Summary:")
                anomaly_counts = anomalies['concept:name'].value_counts()

                fig, ax = plt.subplots()
                anomaly_counts.plot(kind='bar', ax=ax)
                ax.set_xlabel('Activity Number')
                ax.set_ylabel('Number of Anomalies')
                ax.set_title('Anomalies Detected per Activity')
                st.pyplot(fig)

            else:
                st.write("No timestamp anomalies detected in the uploaded event log.")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
if __name__ == "__main__":
    main()

