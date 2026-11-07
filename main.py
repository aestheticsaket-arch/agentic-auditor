from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
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

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="The name of the vendor")
    subtotal: float = Field(description="The subtotal amount before taxes")
    tax_amount: float = Field(description="The total tax amount")
    grand_total: float = Field(description="The final grand total")

# ==========================================
# 2. THE EXTRACTOR AGENT (NODE 1)
# ==========================================
def extractor_agent(state: AuditorState) -> AuditorState:
    print("\n-> [Extractor Agent] Reading document...")
    
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=0)
    structured_llm = llm.with_structured_output(InvoiceData)
    
    prompt = f"Extract the financial data from this document:\n\n{state['raw_text']}"
    
    # If routed back here due to an error, we inject the Critic's reprimand
    if state.get("critic_feedback"):
        prompt += f"\n\nCRITICAL WARNING: The previous extraction failed. {state['critic_feedback']} You must fix the extracted numbers so the math is correct based on the document's logic."
        
    result = structured_llm.invoke(prompt)
    print(f"   [Extractor Output]: {result.model_dump()}")
    
    # Send to critic for review
    return {"extracted_data": result.model_dump(), "status": "pending_review"}

# ==========================================
# 3. THE CRITIC AGENT (NODE 2)
# ==========================================
def critic_agent(state: AuditorState) -> AuditorState:
    print("-> [Critic Agent] Verifying math...")
    data = state["extracted_data"]
    
    subtotal = data.get("subtotal", 0.0)
    tax = data.get("tax_amount", 0.0)
    total = data.get("grand_total", 0.0)
    
    # Perform a hard mathematical check
    calculated_total = subtotal + tax
    
    if abs(calculated_total - total) > 0.1:
        error_msg = f"Math mismatch! Extractor said Grand Total is {total}, but {subtotal} + {tax} = {calculated_total}."
        print(f"   [Critic Result]: FAILED - {error_msg}")
        return {"status": "error", "critic_feedback": error_msg}
    
    print("   [Critic Result]: PASSED - Math is verified.")
    return {"status": "passed", "critic_feedback": ""}

# ==========================================
# 4. THE CONDITIONAL ROUTER (THE LOOP)
# ==========================================
def router(state: AuditorState) -> str:
    if state["status"] == "error":
        return "extractor"  # Loop backwards to try again
    return "end"            # Go to END if passed

# ==========================================
# 5. BUILD THE LANGGRAPH
# ==========================================
workflow = StateGraph(AuditorState)

# Add our two agents
workflow.add_node("extractor", extractor_agent)
workflow.add_node("critic", critic_agent)

# Define the flow
workflow.add_edge(START, "extractor")
workflow.add_edge("extractor", "critic")

# Add the smart routing logic
workflow.add_conditional_edges(
    "critic", 
    router, 
    {"extractor": "extractor", "end": END}
)

# Compile the graph into a runnable application
app = workflow.compile()

# ==========================================
# 6. RUN THE AGENTIC LOOP
# ==========================================
if __name__ == "__main__":
    # We injected a deliberate typo: 50k + 9k should be 59k, but the text says 65k.
    bad_pdf_text = """
    Invoice from TechHardware India Pvt Ltd.
    Date: 04-Nov-2026
    Subtotal: 50000.00
    GST (18%): 9000.00
    Total Payable: 65000.00 
    """
    
    initial_state = AuditorState(
        raw_text=bad_pdf_text,
        extracted_data={},
        critic_feedback="",
        status=""
    )
    
    print("=== STARTING AUDIT ===")
    app.invoke(initial_state)
    print("=== AUDIT COMPLETE ===")