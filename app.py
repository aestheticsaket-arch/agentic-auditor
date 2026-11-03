import streamlit as st
import uuid
import fitz  # PyMuPDF
import pandas as pd
import time
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. STATE & SCHEMA
# ==========================================
class AuditorState(TypedDict):
    raw_text: str
    extracted_data: dict
    critic_feedback: str
    status: str
    error_count: int

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Vendor name")
    subtotal: float = Field(description="Subtotal before taxes")
    tax_amount: float = Field(description="Tax amount")
    grand_total: float = Field(description="Final grand total")

# ==========================================
# 2. AGENTS & GRAPH LOGIC
# ==========================================
def extractor_agent(state: AuditorState):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=0)
    structured_llm = llm.with_structured_output(InvoiceData)
    prompt = f"Extract financial data:\n\n{state['raw_text']}"
    if state.get("critic_feedback"):
        prompt += f"\n\nERROR TO FIX: {state['critic_feedback']}"
    result = structured_llm.invoke(prompt)
    current_errors = state.get("error_count", 0)
    if state.get("status") == "error":
        current_errors += 1
    return {"extracted_data": result.model_dump(), "status": "pending_review", "error_count": current_errors}

def critic_agent(state: AuditorState):
    data = state["extracted_data"]
    sub = data.get("subtotal", 0.0)
    tax = data.get("tax_amount", 0.0)
    total = data.get("grand_total", 0.0)
    if abs((sub + tax) - total) > 0.1:
        return {"status": "error", "critic_feedback": f"Math mismatch: {sub} + {tax} != {total}"}
    return {"status": "passed", "critic_feedback": ""}

def human_review_node(state: AuditorState):
    return {"status": "human_approved"}

def router(state: AuditorState) -> str:
    if state["status"] == "passed":
        return END
    if state["error_count"] >= 2:  
        return "human_review_node"
    return "extractor"

@st.cache_resource
def get_graph():
    workflow = StateGraph(AuditorState)
    workflow.add_node("extractor", extractor_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "critic")
    workflow.add_conditional_edges("critic", router)
    workflow.add_edge("human_review_node", END)
    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["human_review_node"])

app = get_graph()

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    return text

# ==========================================
# 3. ENTERPRISE BATCH QUEUE UI
# ==========================================
st.set_page_config(page_title="FinAudit Pro Batch", page_icon="📊", layout="wide")

# Initialize the queue variables in session state
if "master_data" not in st.session_state:
    st.session_state.master_data = []      
if "current_index" not in st.session_state:
    st.session_state.current_index = 0     
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "batch_started" not in st.session_state:
    st.session_state.batch_started = False

st.title("📊 Enterprise Batch Invoice Processing")

uploaded_files = st.file_uploader("Upload Invoice Batch (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Require a button click to start the engine
    if st.button("Start Batch Audit", type="primary") or st.session_state.batch_started:
        st.session_state.batch_started = True
        
        if st.session_state.current_index < len(uploaded_files):
            current_file = uploaded_files[st.session_state.current_index]
            st.subheader(f"⚙️ Processing File {st.session_state.current_index + 1} of {len(uploaded_files)}: {current_file.name}")
            
            thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
            current_state = app.get_state(thread_config)
            
            # Scenario A: Graph hasn't started for this file yet
            if not current_state or not current_state.values:
                # Add a 2.5-second buffer to respect Groq's RPM limit
                time.sleep(2.5)
                
                raw_pdf_text = extract_text_from_pdf(current_file)
                initial_state = {"raw_text": raw_pdf_text, "extracted_data": {}, "critic_feedback": "", "status": "", "error_count": 0}
                
                try:
                    with st.spinner("Agentic loop running (Rate limit protected)..."):
                        for event in app.stream(initial_state, thread_config):
                            pass
                    st.rerun()
                except Exception as e:
                    # Catch the 429 error and pause instead of crashing the app
                    st.error(f"API Error Detected: {e}")
                    st.warning("Groq rate limit hit. Waiting 10 seconds to cool down before retrying...")
                    time.sleep(10)
                    st.rerun()
                    
            # Scenario B: Math Failed. Execution paused for Human Review
            elif current_state.next == ("human_review_node",):
                st.error(f"⚠️ Anomaly Detected in {current_file.name}: Manual Review Required")
                st.warning(f"**Critic Engine Note:** {current_state.values.get('critic_feedback')}")
                
                current_data = current_state.values.get("extracted_data", {})
                new_sub = st.number_input("Correct Subtotal", value=current_data.get("subtotal", 0.0))
                new_tax = st.number_input("Correct Tax", value=current_data.get("tax_amount", 0.0))
                new_total = st.number_input("Correct Grand Total", value=current_data.get("grand_total", 0.0))
                
                if st.button("Approve & Continue Batch Queue", type="primary"):
                    corrected_data = current_data.copy()
                    corrected_data["subtotal"], corrected_data["tax_amount"], corrected_data["grand_total"] = new_sub, new_tax, new_total
                    app.update_state(thread_config, {"extracted_data": corrected_data})
                    
                    for event in app.stream(None, thread_config):
                        pass
                    st.rerun() 
                    
            # Scenario C: Graph completed successfully
            elif not current_state.next and current_state.values.get("status") in ["passed", "human_approved"]:
                final_data = current_state.values.get("extracted_data")
                final_data["source_file"] = current_file.name
                st.session_state.master_data.append(final_data)
                
                st.session_state.current_index += 1
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
                
        # Batch is finished
        else:
            st.success(f"✅ Batch Processing Complete! {len(uploaded_files)} invoices successfully audited.")
            df = pd.DataFrame(st.session_state.master_data)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Batch to ERP (CSV)",
                data=csv,
                file_name="verified_batch_invoices.csv",
                mime="text/csv",
            )