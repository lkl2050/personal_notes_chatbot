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

from functools import lru_cache

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.prompts import ChatMessage, MessageRole
import logging


class MyObsidianReader(BaseReader):
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def my_load_data(self):
        docs = []
        try:
            print(f"Scanning directory: {self.input_dir}")
            for dirpath, dirnames, filenames in os.walk(self.input_dir):
                # Skip certain directories
                if "Images_Media" in dirnames:
                    dirnames.remove("Images_Media")
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

                for filename in filenames:
                    if filename.endswith(".md"):
                        filepath = os.path.join(dirpath, filename)
                        try:
                            content = MarkdownReader().load_data(Path(filepath))
                            if content:
                                # Validate each document
                                for doc in content:
                                    if doc.text and len(doc.text.strip()) > 0:
                                        docs.append(doc)
                                    else:
                                        print(f"Skipping empty document from {filepath}")
                        except Exception as e:
                            print(f"Error loading file {filepath}: {str(e)}")
                            continue

            print(f"Successfully loaded {len(docs)} valid documents")
            return docs
        except Exception as e:
            print(f"Error in my_load_data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

class ObsidianProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @staticmethod
    def create_enhanced_index(documents: List[Document]):
        try:
            if not documents:
                print("Error: No documents provided")
                return None

            # Add content validation
            valid_documents = []
            for doc in documents:
                if doc.text and len(doc.text.strip()) > 0:
                    valid_documents.append(doc)
                else:
                    print(f"Skipping empty document")

            if not valid_documents:
                print("Error: No valid documents after filtering empty ones")
                return None

            print(f"Processing {len(valid_documents)} valid documents...")

            processor = ObsidianProcessor(chunk_size=512, chunk_overlap=50)
            processed_docs = processor.process_documents(valid_documents)

            if not processed_docs:
                print("Error: No documents after processing")
                return None

            # Validate processed documents
            valid_processed_docs = []
            for doc in processed_docs:
                if doc.text and len(doc.text.strip()) > 0:
                    valid_processed_docs.append(doc)
                else:
                    print(f"Skipping processed document with empty content")

            if not valid_processed_docs:
                print("Error: No valid documents after content validation")
                return None

            print(f"Creating index from {len(valid_processed_docs)} validated documents...")

            index = VectorStoreIndex.from_documents(
                valid_processed_docs,
                show_progress=False
            )

            return index

        except Exception as e:
            print(f"Error creating index: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def clean_text(self, text: str) -> str:
        """Clean Obsidian-specific markdown and formatting"""
        if not text or not isinstance(text, str):
            return ""

        # Remove Obsidian internal links [[...]]
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove empty lines
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        # Remove markdown formatting that might cause issues
        text = re.sub(r'[#*`~]', '', text)
        return text.strip()

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and chunk documents"""
        cleaned_docs = []
        for doc in documents:
            if not doc or not hasattr(doc, 'text'):
                print(f"Skipping invalid document")
                continue

            cleaned_text = self.clean_text(doc.text)
            if cleaned_text:  # Only include if there's actual content
                doc.text = cleaned_text
                cleaned_docs.append(doc)
            else:
                print(f"Skipping document with no content after cleaning")

        if not cleaned_docs:
            print("No valid documents after cleaning")
            return []

        # Convert nodes back to documents with validation
        nodes = self.node_parser.get_nodes_from_documents(cleaned_docs)
        valid_nodes = [node for node in nodes if node.text and len(node.text.strip()) > 0]

        return [Document(text=node.text) for node in valid_nodes]


class PersonalObsidianChat:
    def __init__(self, index):
        self.index = index
        self.memory = []
        self.state = {
            "current_topic": None,
            "last_query_time": None,
            "interaction_count": 0
        }

        self.retriever = self.index.as_retriever(
            similarity_top_k=3
        )

        # Create a more focused query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize",  # Changed response mode
            streaming=False
        )


    def _group_similar_results(self, results: list, similarity_threshold: float = 0.8) -> list:
        """
        Groups semantically similar search results together.

        Args:
            results: List of search results (NodeWithScore objects)
            similarity_threshold: Threshold for grouping similar results (default 0.8)

        Returns:
            list: Grouped results, where each group contains similar content
        """
        if not results:
            return []

        try:
            # Extract embeddings and texts
            embeddings = []
            texts = []
            scores = []

            for result in results:
                # Handle both NodeWithScore and regular Node objects
                text = result.node.text if hasattr(result, 'node') else result.text
                score = result.score if hasattr(result, 'score') else None

                # Get embedding for the text using the same embedding model
                embedding = Settings.embed_model.get_text_embedding(text)

                embeddings.append(embedding)
                texts.append(text)
                scores.append(score)

            # Calculate similarity matrix
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)

            # Group similar results
            grouped_results = []
            used_indices = set()

            for i in range(len(results)):
                if i in used_indices:
                    continue

                # Find similar results
                similar_indices = [
                    j for j in range(len(results))
                    if j not in used_indices and similarity_matrix[i][j] >= similarity_threshold
                ]

                # Create group
                group = {
                    'main_result': {
                        'text': texts[i],
                        'score': scores[i]
                    },
                    'similar_results': [
                        {
                            'text': texts[j],
                            'score': scores[j]
                        }
                        for j in similar_indices if j != i
                    ],
                    'group_score': np.mean([scores[j] for j in similar_indices + [i] if scores[j] is not None])
                }

                grouped_results.append(group)
                used_indices.update(similar_indices + [i])

            # Sort groups by group score
            grouped_results.sort(key=lambda x: x['group_score'], reverse=True)

            return grouped_results

        except Exception as e:
            logger.error(f"Error grouping results: {str(e)}")
            # If grouping fails, return original results in a simple format
            return [{'main_result': {'text': r.node.text if hasattr(r, 'node') else r.text,
                    'score': getattr(r, 'score', None)}}
                    for r in results]

    def _get_relevant_context(self, query: str, conversation_history: list) -> str:
        """
        Retrieves and combines relevant context from both the document index
        and conversation history.

        Args:
            query (str): Current user query
            conversation_history (list): List of previous interactions

        Returns:
            str: Combined relevant context
        """
        # Get relevant documents from index
        doc_nodes = self.retriever.retrieve(query)

        # Extract text from nodes
        doc_context = "\n".join([
            node.node.text if hasattr(node, 'node') else node.text
            for node in doc_nodes
        ])

        # Get recent conversation context
        conv_context = ""
        if conversation_history:
            # Get last 3 exchanges
            recent_history = conversation_history[-6:]
            conv_context = "\n".join([
                f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])

        # Combine contexts with weighting
        combined_context = f"""
        Recent Conversation:
        {conv_context}

        Relevant Documents:
        {doc_context}
        """

        # Optionally trim to fit context window
        max_context_length = 2048  # Adjust based on your model
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length]

        return combined_context

    def chat(self, query: str) -> str:
        try:
            # Update conversation history
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            self.conversation_history.append({"role": "user", "content": query})

            # Get combined context
            context = self._get_relevant_context(query, self.conversation_history)

            # Generate response using context
            response = self.query_engine.query(
                f"""
                Based on the following context:
                {context}

                Question: {query}
                """).response

            # Update conversation history with response
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return "I encountered an error processing your request. Please try again."

    def query(self, query: str) -> str:
        try:
            response = self.query_engine.query(query)

            # Validate response quality
            if len(response.response.strip()) < 10:
                return "I apologize, but I couldn't generate a meaningful response. Could you rephrase your question?"

            # Add confidence score
            confidence = getattr(response, 'score', None)
            if confidence and confidence < 0.5:
                return f"Note: Low confidence response ({confidence:.2f})\n{response.response}"

            return response.response
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return "I encountered an error processing your query. Please try again."

    def search_notes(self, query: str, filters: dict = None) -> list:
        try:
            # Get search results
            results = self.retriever.retrieve(query)

            # Group similar results
            grouped_results = self._group_similar_results(results)

            # Format for display
            formatted_results = []
            for group in grouped_results:
                formatted_group = {
                    'main_text': group['main_result']['text'][:300] + "...",  # Truncate long texts
                    'relevance_score': f"{group['group_score']:.2f}",
                    'similar_count': len(group['similar_results']),
                    'similar_texts': [r['text'][:100] + "..." for r in group['similar_results']]
                }
                formatted_results.append(formatted_group)

            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
