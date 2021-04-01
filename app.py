from transformers import Pipeline
from transformers import pipeline
import streamlit as st

def get_qa_pipeline() -> Pipeline:
    qa = pipeline("question-answering")
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
    
    st.markdown(
		"""
		<style>
		.reportview-container {
		background: url("https://i.pinimg.com/originals/a5/91/17/a59117a046cbc0082afe2ce27622c0c4.jpg")
		}
		</style>
		""", unsafe_allow_html=True) 
	st.markdown(hide_streamlit_style, unsafe_allow_html=True)
       
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
        with st.beta_expander("See prediction details"):
          a=answer["score"]
          b = round(a,3)
          st.write("Confidence:",b*100,"%")
          st.write("Start index:",answer["start"])
          st.write("End index:", answer["end"])
      except:
        st.write("Prediction error!")
