import os
import re
import torch
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from time import perf_counter as timer
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
import fitz  # PyMuPDF

device = "cuda" if torch.cuda.is_available() else "cpu"

nlp = English()
nlp.add_pipe("sentencizer")

# Initialize embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)

# Constants
PDF_PATH = "MC.pdf"
EMBEDDINGS_SAVE_PATH = "text_chunks_and_embeddings.csv"
NUM_SENTENCES_PER_CHUNK = 10  # Number of sentences per text chunk
MIN_TOKEN_LENGTH = 30  # Minimum token length for chunks


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: str) -> list:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    # Adjust page numbers since the PDF might start with preliminary pages
    page_offset = 11  # Update this based on your PDF structure

    for page_number, page in tqdm(enumerate(doc), desc="Reading PDF pages"):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number - page_offset,
            "text": text
        })
    return pages_and_texts


def split_sentences(pages_and_texts: list) -> list:
    """Splits text into sentences using spaCy's sentencizer."""
    for item in tqdm(pages_and_texts, desc="Splitting sentences"):
        doc = nlp(item["text"])
        item["sentences"] = [str(sentence).strip() for sentence in doc.sents]
    return pages_and_texts


def split_into_chunks(pages_and_texts: list, num_sentences_per_chunk: int) -> list:
    """Splits sentences into chunks of a specified size."""
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Creating text chunks"):
        sentences = item["sentences"]
        # Split sentences into chunks
        for i in range(0, len(sentences), num_sentences_per_chunk):
            chunk_sentences = sentences[i:i + num_sentences_per_chunk]
            chunk_text = re.sub(r'\.([A-Z])', r'. \1', " ".join(chunk_sentences))
            chunk_text = chunk_text.replace("  ", " ").strip()
            # Skip chunks that are too short
            if len(chunk_text) / 4 > MIN_TOKEN_LENGTH:
                pages_and_chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": chunk_text
                })
    return pages_and_chunks


def create_embeddings(pages_and_chunks: list) -> tuple:
    """Creates embeddings for each text chunk."""
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
    embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
    return embeddings, pages_and_chunks


def save_embeddings(pages_and_chunks: list, embeddings, save_path: str):
    """Saves the embeddings and chunks to a CSV file."""
    df = pd.DataFrame(pages_and_chunks)
    # Convert embeddings to list for saving
    df["embedding"] = [emb.cpu().numpy().tolist() for emb in embeddings]
    df.to_csv(save_path, index=False)


def load_embeddings(load_path: str) -> tuple:
    """Loads embeddings and text chunks from a CSV file."""
    df = pd.read_csv(load_path)
    # Convert embedding strings back to numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = torch.tensor(np.vstack(df["embedding"].values), dtype=torch.float32).to(device)
    return embeddings, pages_and_chunks


def retrieve_relevant_resources(query: str, embeddings, n_results: int = 5) -> tuple:
    """Retrieves relevant resources for a query based on embeddings."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=n_results)
    return scores.cpu().numpy(), indices.cpu().numpy()


def prompt_formatter(query: str, context_items: list, base_prompt: str) -> str:
    """Formats the prompt by injecting context items and the query."""
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    formatted_prompt = base_prompt.format(context=context, query=query)
    return formatted_prompt


def generate_answer(prompt: str, max_new_tokens: int = 1536, temperature: float = 0.7) -> str:
    """Generates an answer from the model based on the prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Post-process output to remove prompt and special tokens
    answer = output_text.replace(prompt, "").strip()
    answer = answer.replace("</think>", "\n" + "="*80 + "\n" + "Answer after thinking:").strip()
    return answer


def print_wrapped(text, wrap_length=80):
    """Prints text with word wrapping."""
    print(textwrap.fill(text, wrap_length))


# Base prompt template (We can do better :)
BASE_PROMPT = """
Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking; only return the answer.
Make sure your answers are as explanatory as possible.

Now use the following context items to answer the user query:
{context}

User query: {query}
Answer:
"""


def main():
    # Step 1: Read and process the PDF
    pages_and_texts = open_and_read_pdf(PDF_PATH)
    pages_and_texts = split_sentences(pages_and_texts)
    pages_and_chunks = split_into_chunks(pages_and_texts, NUM_SENTENCES_PER_CHUNK)

    # Step 2: Create or load embeddings
    if os.path.exists(EMBEDDINGS_SAVE_PATH):
        embeddings, pages_and_chunks = load_embeddings(EMBEDDINGS_SAVE_PATH)
        print("Loaded embeddings from file.")
    else:
        embeddings, pages_and_chunks = create_embeddings(pages_and_chunks)
        save_embeddings(pages_and_chunks, embeddings, EMBEDDINGS_SAVE_PATH)
        print("Created and saved embeddings.")
    
    # **Loop to accept multiple queries until the user decides to exit**
    while True:
        # Step 3: Get user query
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ('exit', 'quit'):
            print("Exiting the program.")
            break  # Exit the loop and terminate the program

        # Step 4: Retrieve relevant resources
        scores, indices = retrieve_relevant_resources(query, embeddings)

        # Step 5: Prepare context items
        context_items = [pages_and_chunks[i] for i in indices]

        # Step 6: Format the prompt
        prompt = prompt_formatter(query, context_items, BASE_PROMPT)

        # Step 7: Generate and print the answer
        answer = generate_answer(prompt)
        print("\nGenerated Answer:")
        print_wrapped(answer)
        print("\n" + "="*80 + "\n")  # Separator between queries


if __name__ == "__main__":
    # Initialize tokenizer and LLM model
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # Model configuration (update as needed)
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    use_quantization = False

    # Set up quantization configuration if needed
    if use_quantization:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        quantization_config = None

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False
    ).to(device)

    main()