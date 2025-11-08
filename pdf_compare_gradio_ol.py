from pathlib import Path
import PyPDF2
import gradio as gr
import os
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

api_key = os.getenv("OPENAI_KEY")
    
if not api_key:
    raise ValueError("OPENAI_KEY is not set in environment variables.")
os.environ["OPENAI_API_KEY"] = api_key
#openai.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

def display_pdf_files(pdf_path1, pdf_path2=None):
    """
    Display one or two PDF files in Gradio UI side by side
    
    Args:
        pdf_path1: Path to the first PDF file
        pdf_path2: Path to the second PDF file (optional)
    
    Returns:
        HTML string with embedded PDF(s) for Gradio to display
    """
    if not pdf_path1:
        return None
    
    # Process first PDF
    if not os.path.exists(pdf_path1):
        raise gr.Error(f"File not found: {pdf_path1}")
    
    if not pdf_path1.lower().endswith('.pdf'):
        raise gr.Error("Please provide a valid PDF file for the first document")
    
    # Read first PDF file and encode it to base64
    with open(pdf_path1, "rb") as f:
        pdf_data1 = f.read()
    
    base64_pdf1 = base64.b64encode(pdf_data1).decode('utf-8')
    
    # If second PDF is provided, process it
    if pdf_path2:
        if not os.path.exists(pdf_path2):
            raise gr.Error(f"File not found: {pdf_path2}")
        
        if not pdf_path2.lower().endswith('.pdf'):
            raise gr.Error("Please provide a valid PDF file for the second document")
        
        # Read second PDF file and encode it to base64
        with open(pdf_path2, "rb") as f:
            pdf_data2 = f.read()
        
        base64_pdf2 = base64.b64encode(pdf_data2).decode('utf-8')
        
        # Create HTML with two PDFs side by side
        pdf_display = f'''
        <div style="display: flex; width: 100%; height: 600px;">
            <iframe 
                src="data:application/pdf;base64,{base64_pdf1}" 
                width="50%" 
                height="600px" 
                type="application/pdf"
                style="border: none;">
            </iframe>
            <iframe 
                src="data:application/pdf;base64,{base64_pdf2}" 
                width="50%" 
                height="600px" 
                type="application/pdf"
                style="border: none;">
            </iframe>
        </div>
        '''
    else:
        # Create HTML with single PDF
        pdf_display = f'''
        <iframe 
            src="data:application/pdf;base64,{base64_pdf1}" 
            width="100%" 
            height="600px" 
            type="application/pdf"
            style="border: none;">
        </iframe>
        '''
    
    return pdf_display
def extract_text_from_pdf(pdf_path: str, documentName: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from all pages of the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a valid PDF
        Exception: For other PDF reading errors
    """
    # Validate the file path
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    extracted_text = "Below data is " +documentName + " data\n\n"
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Add page separator for better readability
                if text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += text
        extracted_text+="End of " + documentName + " data. \n"
        return extracted_text.strip()
        
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")

def compare_files(pdf_file1, pdf_file2, comparison_question):
        # Extract file paths from uploaded file objects
        pdf_path1 = pdf_file1.name if pdf_file1 else None
        pdf_path2 = pdf_file2.name if pdf_file2 else None
        
        if not pdf_path1 or not pdf_path2:
            raise gr.Error("Both PDF files are required for comparison")
        # Show loading message
        loading_html = '''
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 20px 0;
            text-align: center;
        ">
            <div style="
                background: white;
                border-radius: 10px;
                padding: 40px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            ">
                <div style="
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #667eea;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                "></div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
                <h2 style="
                    color: #667eea;
                    margin-top: 20px;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">🔄 Comparing Documents...</h2>
                <p style="
                    color: #666;
                    font-size: 14px;
                    margin-top: 10px;
                ">Please wait while we analyze the documents</p>
            </div>
        </div>
        '''
        yield loading_html
        
        doc1_text = extract_text_from_pdf(pdf_path1,"First document")
        doc2_text = extract_text_from_pdf(pdf_path2,"Second document")

        retriever=doc1_text + "\n" + doc2_text

        template = """
        You are an expert document comparison AI. Your task is to compare the data
        from two documents provided in the context below. The documents are named in their metadata.

        **Comparison Query:** {question}

        **Instructions:**
        1.  Carefully analyze the content related to the query from both documents.
        2.  Identify and list all similarities and differences clearly, citing which document (Document 1 or Document 2, based on the source metadata) each piece of information comes from.
        3.  Format the output with separate sections for Similarities and Differences.

        **Context from Documents:**
        {context}
        """
        comparison_prompt = ChatPromptTemplate.from_template(template)
        # Create the RAG chain using LCEL (LangChain Expression Language)
        rag_chain = (
            {"context": lambda x: retriever, "question": RunnablePassthrough()}
            | comparison_prompt
            | llm
            | StrOutputParser()
        )
        result = rag_chain.invoke(comparison_question)
        
        # Format the result with nice HTML styling - preserving plain text formatting
        result_display = f'''
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 20px 0;
        ">
            <div style="
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            ">
                <h2 style="
                    color: #667eea;
                    margin-top: 0;
                    padding-bottom: 15px;
                    border-bottom: 3px solid #667eea;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">📊 Document Comparison Results</h2>
                <pre style="
                    color: #333;
                    line-height: 1.6;
                    font-size: 14px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 5px solid #764ba2;
                    margin-top: 15px;
                    overflow-x: auto;
                ">{result}</pre>
            </div>
        </div>
        '''
        
        yield result_display

def load_files(pdf_file1, pdf_file2):
    """
    Load uploaded PDFs and make compare button and prompt dropdown visible.
    Gradio File component automatically uploads files to server temp directory.
    
    Args:
        pdf_file1: Uploaded PDF file object (contains .name attribute with server path)
        pdf_file2: Uploaded PDF file object (contains .name attribute with server path)
    """
    # Extract the server-side file paths from the uploaded file objects
    pdf_path1 = pdf_file1.name if pdf_file1 else None
    pdf_path2 = pdf_file2.name if pdf_file2 else None
    
    if not pdf_path1:
        raise gr.Error("Please upload the first PDF file")
    if not pdf_path2:
        raise gr.Error("Please upload the second PDF file")
    
    # Display the uploaded PDFs
    pdf_html = display_pdf_files(pdf_path1, pdf_path2)
    return pdf_html, gr.update(visible=True), gr.update(visible=True, interactive=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    
def update_prompt_text(selected_choice):
    """Update prompt text based on dropdown selection"""
    if selected_choice == "Custom prompt":
        return ""
    elif selected_choice == "Financial or Accounting Documents":
        return """Compare First document (Original Financial Report/Invoice) and Second document (Revised Financial Report/Invoice). Focus on all numerical and transactional differences. Identify:
                 1. Changes in total amounts, line items, or percentages.
                 2. Differences in account numbers, dates, or vendor/client information.
                 3. Any missing or newly added sections (e.g., a new expense category). Present the output as a table showing the field, First document value, and Second document value."""
    elif selected_choice == "Financial or Accounting Documents transaction differences":
        return """Task: Compare two financial documents — Document 1 (Original Financial Report/Invoice) and Document 2 (Revised Financial Report/Invoice) — and generate a comprehensive structured summary of all differences.
                    
                    Focus Areas:
                    1. Numerical and Transactional Changes:
                        . Variations in totals, subtotals, taxes, and line item amounts.
                        . Adjustments in rates, quantities, or percentages.
                        . Any rounding or recalculation discrepancies.
                    2. Data and Metadata Differences:
                        . Modifications in account numbers, invoice numbers, or reference IDs.
                        . Changes in issue or due dates.
                        . Alterations in vendor, client, or payer details.
                    3. Structural Variations:
                        . Missing, removed, or newly added line items or sections (e.g., a new expense or revenue category).
                        . Reordered sections or renamed headers.
                    
                    Output Format:
                    Present the comparison in a structured table with the following columns:
                    | Field / Section | Document 1 (Original) | Document 2 (Revised) | Type of Change | Description / Remarks |

                    Additional Requirements:
                     . Highlight any financial impact (increase/decrease) in totals or balances.
                     . Provide a short summary paragraph explaining the overall trend (e.g., “Total payable increased by 8% due to additional line items in marketing expenses.”).
                     . If sections are identical, summarize them as “No change detected.”
                     . Ensure accuracy in numerical interpretation, even if formats differ (e.g., “$1,200.00” vs “1200”)."""
    elif selected_choice == "Legal or Contractual Documents":
        return """Compare First Document (Original Contract) and Second Document (Revised Contract). Identify all substantive differences, especially focusing on changes to clauses, definitions, dates, party names, terms, or conditions. Present the differences in a bulleted list, clearly stating:
                  1. The clause/section where the change occurred.
                  2. The original text from First Document.
                  3. The new text from Second Document."""   
    elif selected_choice == "Health or Medical Records (e.g., Lab Reports, Clinical Notes)":
        return """Compare First Document (Initial Health Record/Lab Result) and Second Document (Updated Health Record/Lab Result). List all differences in critical medical data and administrative information. Specifically look for changes in:
                  1. Diagnosis codes (ICD codes), or the stated primary diagnosis.
                  2. Medication names, dosages, or frequency.
                  3. Lab results (e.g., a change in a specific test value or status).
                  4. Patient demographics like date of birth or insurance ID. Structure the output clearly under two sections: 'Clinical Differences' and 'Administrative Differences'.""" 
    else:
        return """Act as a professional document analyst. Your task is to compare two versions of a document: First Document (The Original) and Second Document (The Revision).
                  Identify all differences between the two documents.
                  Focus primarily on substantive changes, including additions, deletions, and modifications to sentences, paragraphs, or lists. Minor formatting changes should be ignored unless they affect clarity.
                  For each difference found, present the information in the following structured format:
                    1. Location: (e.g., Page X, Paragraph Y, or Section Title)
                    2. Original Text (First Document): [Quote the text that was in First Document, which is now changed or deleted.]
                    3. Revised Text (Second Document): [Quote the new or modified text that appears in Second Document.]
                    4. Change Type: (Addition, Deletion, or Modification)
                  Conclude with a brief summary (2-3 sentences) detailing the most significant overall change made in Second Document."""

#C:\Users\sanisett\AppData\Local\Temp\gradio
# Create Gradio interface
with gr.Blocks(title="PDF Viewer") as demo:
    gr.Markdown("# PDF Viewer Application")
    gr.Markdown("Enter the path to a PDF file to display it below")
    
    with gr.Row():
        pdf_input_file1 = gr.File(
            label="Upload First PDF Document",
            file_types=[".pdf"],
            type="filepath",
            scale=3
        )
        pdf_input_file2 = gr.File(
            label="Upload Second PDF Document",
            file_types=[".pdf"],
            type="filepath",
            scale=3
        )
        display_btn = gr.Button("Load Documents", variant="primary", scale=1)
    pdf_viewer = gr.HTML(label="PDF Viewer")
    
    prompt_label = gr.Markdown("### Select Prompt", visible=False)
    with gr.Row():
        prompt_dropdown = gr.Dropdown(
            choices=[
                "Financial or Accounting Documents transaction differences",
                "Financial or Accounting Documents",
                "Legal or Contractual Documents",
                "Health or Medical Records (e.g., Lab Reports, Clinical Notes)",
                "General Documents",
                "Custom prompt"
            ],
            label="Document Type",
            value="General Documents",
            visible=False,
            interactive=True,
            scale=3
        )
        prompt_text = gr.Textbox(
            label="Prompt text",
            placeholder="Enter your custom prompt here...",
            value=update_prompt_text("General Documents"),
            visible=False,
            interactive=True,
            scale=3,
            lines=4
        )
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[
                "Microsoft-phi (local)",
                "gemma (local)",
                "gpt-4",
                "gpt-4o-mini",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo",
                "gpt-4o",
                "gemini-1.5-flash"
            ],
            label="Select Model",
            value="gpt-4o-mini",
            visible=False,
            interactive=True,
            scale=3
        )
        compare_btn = gr.Button("Compare Documents", variant="primary", scale=1, visible=False)
    pdf_compare_result_view = gr.HTML(label="Document Comparison Result")
    
    display_btn.click(
        fn=load_files,
        inputs=[pdf_input_file1, pdf_input_file2],
        outputs=[pdf_viewer, prompt_label, prompt_dropdown,prompt_text, model_dropdown, compare_btn]
    )
    compare_btn.click(
        fn=compare_files,
        inputs=[pdf_input_file1, pdf_input_file2, prompt_text],
        outputs=[pdf_compare_result_view]
    )
    prompt_dropdown.change(
        fn=update_prompt_text,
        inputs=[prompt_dropdown],
        outputs=[prompt_text]
    )
    
    # Example section
    gr.Markdown("---")
    gr.Markdown("### Instructions:")
    gr.Markdown("- Upload your PDF files using the file upload buttons")
    gr.Markdown("- Click 'Load Documents' to display them in the viewer")
    gr.Markdown("- Select a document type or enter a custom prompt")
    gr.Markdown("- Click 'Compare Documents' to analyze the differences")

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="OTX-FYCKLX3",
        server_port=7860
    )
