import sys
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,  BitsAndBytesConfig, AutoModel
import torch

# Load the local Microsoft Phi model
local_model_path = r"../models/gemma-3-4b-it"
print(f"Loading model from {local_model_path}...")

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#device = torch.device("cuda:1")
#     print(f"GPU available: {torch.cuda.get_device_name(0)}")
#     print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, using CPU")
device = torch.device("cpu")
if os.path.exists(local_model_path):
    print("Loading model from local directory...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True )
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True
    )
    #for 4-bit quantization, uncomment below and comment above
    # quantization_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True
    # )
    #model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto",quantization_config=quantization_config, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map={"": "cpu"}, torch_dtype=torch.float16, local_files_only=True)
    # Note: Do NOT call model.to(device) when using device_map="auto" with quantization
    # The model is already properly dispatched by device_map="auto"
    #model.to(device)
else:
    print("Local model not found. Downloading from Hugging Face...")
    sys.exit(1)  # Exit with error code 1

# Create a text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    top_p=0.95,
    return_full_text=False
)

# Wrap the pipeline in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("Model loaded successfully!")

# Load PDF 1
loader_1 = PyPDFLoader(r"C:\Work_Learnings\Learning\AI\TestData\Account.pdf")
docs_1 = loader_1.load()

# Unique metadata for First Document
metadata_1 = {
    "doc_id": "Document_A_Policy_v1",
    "source_title": "First Document (The Original)",
    "version": "v1.0"
}

# Inject metadata into all Document objects for A
for doc in docs_1:
    # This combines the loader's default metadata (like 'page') with your custom data
    doc.metadata.update(metadata_1)

# Load PDF 2
loader_2 = PyPDFLoader(r"C:\Work_Learnings\Learning\AI\TestData\Account_bigchange.pdf")
docs_2 = loader_2.load()

metadata_2 = {
    "doc_id": "Document_B_Policy_v2",
    "source_title": "Second document",
    "version": "v2.0"
}

# Inject metadata into all Document objects for B
for doc in docs_2:
    doc.metadata.update(metadata_2)

# Combine documents
all_docs = docs_1 + docs_2

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(all_docs)

# Create embeddings and vector store using local embeddings
print("Creating embeddings...")
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     #model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
#     model_kwargs={'device':'cpu'}
# )
# ---- Custom Gemma Embeddings class ----
class GemmaEmbeddings(Embeddings):
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or local_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Gemma model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # shape: [batch, seq_len, hidden_dim]
        emb = last_hidden.mean(dim=1)  # mean pooling
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # normalize
        return emb.squeeze().cpu().numpy()

    def embed_query(self, text):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

embeddings = GemmaEmbeddings(model_path=r"../models/gemma-3-4b-it")
vectorstore = Chroma.from_documents(split_docs, embeddings)
print("Vector store created!")

# Create retriever
retriever = vectorstore.as_retriever()

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


# Function to format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def safe_truncate(text, max_chars=6000):
    return text[:max_chars]

# Create the RAG chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs | (lambda x: safe_truncate(x)), "question": RunnablePassthrough()}
    | comparison_prompt
    | llm
    | StrOutputParser()
)

# Run the comparison
if __name__ == "__main__":
    # Define the comparison question
   # comparison_question = "Compare these two documents. What are the key similarities and differences?"
    comparison_question = """Compare First document (Original Financial Report/Invoice) and Second document (Revised Financial Report/Invoice). Focus on all numerical and transactional differences. Identify:
                     Changes in total amounts, line items, or percentages.
                     Differences in account numbers, dates, or vendor/client information.
                     Any missing or newly added sections (e.g., a new expense category). Present the output as a table showing the field, First document value, and Second document value."""
    # Invoke the chain
    print("Starting PDF comparison...\n")
    print("=" * 80)
    result = rag_chain.invoke(comparison_question)
    print(result)
    print("=" * 80)