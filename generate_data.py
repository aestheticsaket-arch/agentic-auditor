import fitz  # PyMuPDF
import random
import os

# Create a folder to hold the batch data
os.makedirs("batch_invoices", exist_ok=True)
vendors = ["TechHardware India", "CloudNet Solutions", "Alpha IT Services", "DataSys Analytics", "Nexus Corp"]

for i in range(1, 51):
    doc = fitz.open()
    page = doc.new_page()
    
    vendor = random.choice(vendors)
    subtotal = round(random.uniform(10000, 90000), 2)
    tax = round(subtotal * 0.18, 2)
    
    # Inject a deliberate math hallucination into 3 specific files to test the LangGraph failsafe
    if i in [12, 34, 47]:
        total = round(subtotal + tax + 5000.00, 2) 
    else:
        total = round(subtotal + tax, 2)
        
    invoice_text = (
        f"Invoice from {vendor} Pvt Ltd.\n"
        f"Date: 05-Nov-2026\n"
        f"Invoice Number: INV-2026-{1000+i}\n\n"
        f"Subtotal: {subtotal}\n"
        f"GST (18%): {tax}\n"
        f"Total Payable: {total}"
    )
    
    # Write the text into the PDF
    page.insert_text(fitz.Point(50, 70), invoice_text, fontsize=12)
    doc.save(f"batch_invoices/invoice_{i:02d}.pdf")
    doc.close()

print("✅ Successfully generated 50 PDF invoices in the 'batch_invoices' folder.")