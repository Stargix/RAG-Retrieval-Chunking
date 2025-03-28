from fixed_token_chunker import FixedTokenChunker
import numpy as np
import torch
import json
from functools import lru_cache
from sentence_transformers import SentenceTransformer

class CachedEncoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    @lru_cache(maxsize=1000)
    def encode_cached(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    
    def encode(self, texts, batch_size=128, convert_to_tensor=True):
        # For single texts, use the cached version
        if isinstance(texts, str):
            return self.encode_cached(texts)
        
        # For lists with single text, use cached version
        if isinstance(texts, list) and len(texts) == 1:
            return torch.stack([self.encode_cached(texts[0])])
        
        # For multiple texts, use the standard batch encoding
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=convert_to_tensor)
    
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, chunks, top_k=3):
        """
        Rerank the provided chunks based on their relevance to the query using a cross-encoder model.

        Args:
            query (str): The input query for which relevance is evaluated.
            chunks (list of str): The list of text chunks to be reranked.
            top_k (int, optional): The number of top relevant chunks to return. Defaults to 3.

        Returns:
            list of str: The top_k most relevant chunks sorted by relevance score.
        """
        # Create pairs of (query, chunk)
        pairs = [[query, chunk] for chunk in chunks]
        # Predict relevance scores for each pair
        scores = self.model.predict(pairs)
        # Sort chunks by descending scores
        ranked_indices = np.argsort(scores)[::-1]
  
        return [chunks[i] for i in ranked_indices[:top_k]]

class Chunker_RAG:
    def __init__(self, text, encoder, reranker, chunk_size=100, chunk_overlap=20):
        self.text = text
        self.encoder = encoder
        self.reranker = reranker
        chunks = self._split_chunks(text, chunk_size, chunk_overlap)
        self.text_chunks = chunks[0]
        self.chunk_ranges = chunks[1]
        self.encoded_chunks = self._embedding_function(self.text_chunks, convert_to_tensor=True)

    def _split_chunks(self, text, chunk_size, overlap):
        """
        Split the input text into overlapping chunks of a specified size.

        Args:
            text (str): The input text to be split into chunks.
            chunk_size (int): The size of each chunk in tokens.
            overlap (int): The number of overlapping tokens between consecutive chunks.

        Returns:
            tuple: A tuple containing:
            - text_chunks (list of str): The list of text chunks.
            - chunk_ranges (list of tuple): The list of start and end indices for each chunk in the original text.
        """
        chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        text_chunks = chunker.split_text(text)

        chunk_ranges = []
        for chunk in text_chunks:
            # Buscar el índice inicial del fragmento en el texto original
            start_index = text.find(chunk)
            end_index = start_index + len(chunk)
            chunk_ranges.append((start_index, end_index))

        return text_chunks, chunk_ranges
    
    def _embedding_function(self, text, batch_size=128, convert_to_tensor=True):
        return self.encoder.encode(text, batch_size=batch_size, convert_to_tensor=convert_to_tensor)

    def _base_retrieve(self, question, quantity_chunks=1):
        """
        Retrieve the most relevant text chunks and their corresponding ranges based on a given question.
        This method computes the cosine similarity between the embedding of the input question and 
        pre-encoded text chunks to identify the most relevant chunks.
        Args:
            question (str): The input question for which relevant chunks are to be retrieved.
            quantity_chunks (int, optional): The number of top relevant chunks to retrieve. Defaults to 1.
        Returns:
            tuple: A tuple containing:
                - retrieved_chunks (list of str): The list of retrieved text chunks.
                - retrieved_ranges (list of tuple): The list of corresponding ranges for the retrieved chunks.
        """

        question_embedding = self._embedding_function([question])
        # Normalize embeddings for cosine similarity
        normalized_encoded_chunks = self.encoded_chunks / torch.norm(self.encoded_chunks, dim=1, keepdim=True)
        normalized_question_embedding = question_embedding / torch.norm(question_embedding, dim=1, keepdim=True)
        # Compute cosine similarities
        similarities = (normalized_encoded_chunks @ normalized_question_embedding.T).squeeze().cpu().numpy()
        most_similar_indices = similarities.argsort()[-quantity_chunks:][::-1]
        retrieved_chunks = [self.text_chunks[idx] for idx in most_similar_indices]
        retrieved_ranges = [self.chunk_ranges[idx] for idx in most_similar_indices]
        return retrieved_chunks, retrieved_ranges
    
    def retrieve_answer(self, question, quantity_chunks=5):
        """
        Retrieve the most relevant answer chunks and their corresponding ranges for a given question.
        Args:
            question (str): The input question for which the answer is to be retrieved.
            quantity_chunks (int, optional): The number of top relevant chunks to retrieve. 
                Defaults to 5.
        Returns:
            tuple: A tuple containing:
                - str: A concatenated string of the top relevant chunks.
                - list: A list of ranges corresponding to the retrieved chunks.
        """
        # 1. Initial retrieval (more chunks than necessary)
        initial_chunks, initial_ranges = self._base_retrieve(question, quantity_chunks*3)
        
        # 2. Reranking of the best candidates
        final_chunks = self.reranker.rerank(question, initial_chunks, top_k=quantity_chunks)
        
        # 3. Obtain final ranges
        final_indices = [initial_chunks.index(chunk) for chunk in final_chunks]
        final_ranges = [initial_ranges[i] for i in final_indices]
        
        return " ".join(final_chunks), final_ranges
    
    def _compute_metrics(self, reference_ranges, retrieved_ranges):
        """
        Calcula las métricas de evaluación: precisión, recall y IoU.
        
        Args:
            reference_ranges: Lista de tuplas (inicio, fin) que representan los rangos relevantes.
            context_ranges: Lista de tuplas (inicio, fin) que representan los rangos recuperados.
        
        Returns:
            tuple: (precision, recall, iou) métricas calculadas.
        """
        # Calculate intersection between reference and retrieved ranges
        intersection = 0
        for (ref_start, ref_end) in reference_ranges:
            for (ret_start, ret_end) in retrieved_ranges:
                overlap_start = max(ref_start, ret_start)
                overlap_end = min(ref_end, ret_end)
                intersection += max(0, overlap_end - overlap_start)
        
        # Total relevant and retrieved tokens
        total_relevant = sum(end - start for start, end in reference_ranges)
        total_retrieved = sum(end - start for start, end in retrieved_ranges)
        
        # Ensure that the intersection does not exceed the total number of relevant tokens
        intersection = min(intersection, total_relevant)
        
        precision = intersection / total_retrieved if total_retrieved > 0 else 0
        recall = intersection / total_relevant if total_relevant > 0 else 0
        union = total_relevant + total_retrieved - intersection
        iou = intersection / union if union > 0 else 0
        
        return precision, recall, iou
    
    def _extract_reference_ranges(self, references):
        """
        Extract reference ranges from a JSON-formatted string.

        Args:
            references (str): A JSON string containing a list of dictionaries with 'start_index' and 'end_index' keys.

        Returns:
            list of tuple: A list of tuples representing the start and end indices of the reference ranges.
        """
        query_list = json.loads(references)
        reference_ranges = [(item['start_index'], item['end_index']) for item in query_list]
        return reference_ranges
    
    def get_metrics(self, questions, references, quantity_chunks=1):
        """
        Calculate precision, recall, and IoU metrics for a list of questions.
        
        Args:
            questions (list): List of question strings to evaluate.
            references (list): List of reference strings in JSON format containing the ground truth ranges.
            quantity_chunks (int, optional): Number of chunks to retrieve per question. Defaults to 1.
            
        Returns:
            tuple: Three lists containing precision, recall, and IoU values for each question.
        """
        precisions = []
        recalls = []
        ious = []
        
        for question, reference in zip(questions, references):
            retrieved_chunks, retrieved_ranges = self.retrieve_answer(question, quantity_chunks)
            reference_ranges = self._extract_reference_ranges(reference)

            precision, recall, iou = self._compute_metrics(reference_ranges, retrieved_ranges)
            precisions.append(precision)
            recalls.append(recall)
            ious.append(iou)
        
        return precisions, recalls, ious
    
    def context_retrieval(self, questions, quantity_chunks=5):
        """
        Retrieve relevant chunks for a list of questions without reranking.
        
        Args:
            questions (list): List of question strings to retrieve context for.
            quantity_chunks (int, optional): Number of chunks to retrieve per question. Defaults to 5.
            
        Returns:
            tuple: Two lists containing retrieved chunks and their corresponding ranges for each question.
        """
        all_retrieved_chunks = []
        all_retrieved_ranges = []
        for question in questions:
            retrieved_chunks, retrieved_ranges = self._base_retrieve(question, quantity_chunks)
            all_retrieved_chunks.append(retrieved_chunks)
            all_retrieved_ranges.append(retrieved_ranges)
        return all_retrieved_chunks, all_retrieved_ranges