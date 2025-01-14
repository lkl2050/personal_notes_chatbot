# %%
from pathlib import Path
import os
import re
from typing import List

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document
)
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.prompts import PromptTemplate, MessageRole

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file.markdown import MarkdownReader

class ObsidianProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def clean_text(self, text: str) -> str:
        """Clean Obsidian-specific markdown and formatting"""
        # Remove Obsidian internal links [[...]]
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove empty lines
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        return text

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and chunk documents"""
        cleaned_docs = []
        for doc in documents:
            if doc.text.strip():  # Skip empty documents
                cleaned_text = self.clean_text(doc.text)
                if cleaned_text:
                    doc.text = cleaned_text
                    cleaned_docs.append(doc)

        return self.node_parser.get_nodes_from_documents(cleaned_docs)

class MyObsidianReader(BaseReader):
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def my_load_data(self):
        docs = []
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            # Skip certain directories
            if "Images_Media" in dirnames:
                dirnames.remove("Images_Media")
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    content = MarkdownReader().load_data(Path(filepath))
                    docs.extend(content)
        return docs



class PersonalObsidianChat:
    def __init__(self, index):
        self.index = index
        self.chat_engine = index.as_chat_engine(
            verbose=False,
            prompt=chat_prompt
        )
        self.query_engine = index.as_query_engine()

    def chat(self, query: str) -> str:
        try:
            response = self.chat_engine.chat(query)
            return response.response
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def query(self, query: str) -> str:
        try:
            response = self.query_engine.query(query)
            return response.response
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def search_notes(self, query: str, top_k: int = 3):
        """Search through notes and return most relevant passages"""
        try:
            print(f"Searching for: {query}")
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)

            if not results:
                print("No results found")
                return []

            print(f"Found {len(results)} results")
            return results

        except Exception as e:
            print(f"Error in search_notes: {str(e)}")
            return []

def init_llm():
    try:
        llm = Ollama(
            model="tinydolphin",
            request_timeout=120.0,
            temperature=0.7,  # Add temperature control
            context_window=2048  # Set context window
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def create_enhanced_index(documents: List[Document]):
    processor = ObsidianProcessor(chunk_size=512, chunk_overlap=50)
    processed_docs = processor.process_documents(documents)

    return VectorStoreIndex.from_documents(
        processed_docs,
        show_progress=True
    )



# %%
from llama_index.core.prompts import ChatMessage, MessageRole

chat_prompt = PromptTemplate(
    template=(
        "You are a helpful AI assistant accessing personal notes. "
        "Provide clear, direct responses based on the provided context. "
        "If no relevant information is found, say so clearly.\n\n"
        "Context: {context}\n"
        "Human: {query}\n"
        "Assistant: "
    )
)

# %%
# Main execution

# Initialize LLM
llm = init_llm()

# Set up global settings
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Load and process documents
reader = MyObsidianReader(input_dir="/path/to/your/obsidian/vault")
documents = reader.my_load_data()

# Create index
index = create_enhanced_index(documents)

# Create chat interface

# obsidian_chat = PersonalObsidianChat(index)
# response = obsidian_chat.chat("What are my notes about social perception?")
# print(response)


# # Interactive chat loop
# while True:
#     query = input("You: ")
#     if query.lower() in ['quit', 'exit']:
#         break
#     response = obsidian_chat.chat(query)
#     print(f"Bot: {response}")




# %%
def chat_interface():
    obsidian_chat = PersonalObsidianChat(index)

    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Search notes")
        print("3. Exit")

        choice = input("Choose an option (1-3): ")

        if choice == "1":
            query = input("Your question: ")
            response = obsidian_chat.chat(query)
            print(f"Bot: {response}\n")

        elif choice == "2":
            search_term = input("Search term: ")
            results = obsidian_chat.search_notes(search_term)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Score: {result.score if hasattr(result, 'score') else 'N/A'}")
                    print(f"Text: {result.text[:200]}...")
                    print("-" * 50)
            else:
                print("No results found")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please try again.")

# Start the interface
chat_interface()


