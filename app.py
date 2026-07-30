# app.py
import gradio as gr
from src.rag.pipeline import ComplaintRAGPipeline
from src.rag.retriever import ComplaintRetriever
from src.rag.generator import ComplaintGenerator
from src.rag.vector_store import ComplaintVectorStore

# -------------------------------
# Load your vector store and RAG
# -------------------------------
vector_store = ComplaintVectorStore(parquet_path="data/complaint_embeddings.parquet")
vector_store.load_or_build()

retriever = ComplaintRetriever(vector_store)
generator = ComplaintGenerator(model_name="google/flan-t5-base")  # better than small

rag_pipeline = ComplaintRAGPipeline(retriever, generator)

# -------------------------------
# Chat function
# -------------------------------
def ask_question(user_input):
    if not user_input.strip():
        return "", ""
    
    result = rag_pipeline.answer(user_input)
    
    # Extract the top 2 sources
    sources_text = ""
    for i, src in enumerate(result["sources"], start=1):
        sources_text += f"Source {i}:\n{src['text']}\n\n"
    
    return result["answer"], sources_text.strip()


# -------------------------------
# Gradio interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## CrediTrust Complaint Assistant")
    gr.Markdown("Ask questions about customer complaints and get answers with sources.")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your Question", placeholder="Type your question here...")
            submit_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=10)
            sources_output = gr.Textbox(label="Sources", interactive=False, lines=10)
    
    submit_btn.click(ask_question, inputs=user_input, outputs=[answer_output, sources_output])
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[answer_output, sources_output])

# -------------------------------
# Launch app
# -------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
