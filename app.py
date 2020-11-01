from transformers import Pipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st

def get_qa_pipeline() -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    qa = pipeline("question-answering", model = model, tokenizer=tokenizer)
    return qa

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question,
        "context": paragraph
    }
    return pipeline(input)

def format_text(paragraph: str, start_idx: int, end_idx: int) -> str:
    return paragraph[:start_idx] + "**" + paragraph[start_idx:end_idx] + "**" + paragraph[end_idx:]

if __name__ == "__main__":
    
       
    st.title("QUESTION ANSWERING")
    st.text("PSG College of Technology - Project Work I")
    
    passage = st.text_area("PASSAGE", "")
    question = st.text_input("QUESTION", "")

    if st.button("Get Answer"):
      pipeline = get_qa_pipeline()
      #st.write(pipeline.model)
      #st.write(pipeline.model.config)
      try:
        answer = answer_question(pipeline, question, passage)
        #start_idx = answer["start"]
        #end_idx = answer["end"]
        st.write("Prediction:", answer["answer"])
      except:
        st.write("Prediction error!")
