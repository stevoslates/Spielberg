import os
import json
import pandas as pd
import openai
from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAGPipeline:
    def __init__(self):
        self.llm_name = "gpt-4-turbo"
        self.llm = ChatOpenAI(model_name=self.llm_name)
        self.embedding = OpenAIEmbeddings()
        self.normal_vectordb = None
        self.selfquery_vectordb = None
        self.normal_retriever = None
        self.selfquery_retriever = None

    def setup_normal_pipeline(self):
        reviews = pd.read_csv("/Users/stevenslater/Desktop/LangChainRag/letterboxeddata/reviews.csv")
        ratings = pd.read_csv("/Users/stevenslater/Desktop/LangChainRag/letterboxeddata/ratings.csv")
        
        reviews = reviews.drop(columns=["Letterboxd URI", "Date", "Tags", "Rewatch", "Watched Date"])
        ratings = ratings.drop(columns=["Letterboxd URI", "Date"])
        
        merged_df = pd.merge(ratings, reviews[['Name', 'Review']], on='Name', how='outer')
        merged_df = merged_df.fillna('')
        merged_df['all'] = merged_df.apply(lambda row: f"Movie Name: {row['Name']}, Rating (Out of 5): {row['Rating']}, Review: {row['Review']}", axis=1)
        
        movies_json = merged_df[['all']].to_json(orient='records')
        with open('merged_data.json', 'w') as f:
            f.write(movies_json)
        
        loader = JSONLoader(file_path='merged_data.json', jq_schema='.[]', content_key="all", text_content=False)
        data = loader.load()
        
        persist_directory = 'docs/chroma/'
        self.normal_vectordb = Chroma.from_documents(documents=data, embedding=self.embedding, persist_directory=persist_directory)
        self.normal_retriever = self.normal_vectordb.as_retriever()

    def setup_selfquery_pipeline(self):
        reviews = pd.read_csv("/Users/stevenslater/Desktop/LangChainRag/letterboxeddata/reviews.csv")
        ratings = pd.read_csv("/Users/stevenslater/Desktop/LangChainRag/letterboxeddata/ratings.csv")
           
        reviews = reviews.drop(columns=["Letterboxd URI", "Date", "Tags", "Rewatch", "Watched Date"])
        ratings = ratings.drop(columns=["Letterboxd URI", "Date"])
        
        merged_df = pd.merge(ratings, reviews[['Name', 'Review']], on='Name', how='outer')
        merged_df = merged_df.fillna('')
        new_df = merged_df.drop(['all'], axis=1, errors='ignore')
        new_df = new_df.to_json(orient='records')
        with open('new_df.json', 'w') as f:
            f.write(new_df)
        
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["Rating"] = record.get("Rating")
            metadata["Review"] = record.get("Review")
            return metadata
        
        loader = JSONLoader(file_path='new_df.json', jq_schema='.[]', content_key="Name", text_content=False, metadata_func=metadata_func)
        data = loader.load()
        
        persist_directory = 'docs/chroma/'
        self.selfquery_vectordb = Chroma.from_documents(documents=data, embedding=self.embedding, persist_directory=persist_directory)
        
        metadata_field_info = [
            AttributeInfo(name="Rating", description="The rating given by the user, it is out of 5, where 5 is the highest and best rating.", type="string"),
            AttributeInfo(name="Review", description="The review given by the user for the movie, if it is good or bad etc, may also say what they liked and disliked about the movie.", type="string"),
        ]
        document_content_description = "Movie Names"
        self.selfquery_retriever = SelfQueryRetriever.from_llm(self.llm, self.selfquery_vectordb, document_content_description, metadata_field_info, verbose=True)

    def run_normal_query(self, question):
        if self.normal_retriever is None:
            self.setup_normal_pipeline()
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.normal_retriever)
        result = qa_chain({"query": question})
        return result["result"]

    def run_selfquery(self, question):
        if self.selfquery_retriever is None:
            self.setup_selfquery_pipeline()

        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.selfquery_retriever, return_source_documents=True)
        result = qa_chain({"query": question})
        return result["result"]

def main():
    pipeline = RAGPipeline()
    choice = input("Select RAG pipeline (normal/selfquery): ").strip().lower()
    if choice not in ["normal", "selfquery"]:
        print("Invalid choice. Please select 'normal' or 'selfquery'.")
        return
    
    while True:
        question = input("Enter your query (or type 'end' to stop): ").strip()
        if question.lower() == "end":
            break
        if choice == "normal":
            response = pipeline.run_normal_query(question)
        elif choice == "selfquery":
            response = pipeline.run_selfquery(question)
        print(response)

if __name__ == "__main__":
    main()
