import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from bert_score import BERTScorer
import torch
from functools import lru_cache
from nltk.translate.bleu_score import sentence_bleu

# Global model instances and caches
SENTENCE_TRANSFORMER = None
BERT_SCORER = None
ABSTRACTIVE_PIPELINE = None
SENTENCE_EMBEDDING_CACHE = {}
ROUGE_SCORER = None

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

@lru_cache(maxsize=1024)
def get_sentence_embedding(sentence):
    """Cache sentence embeddings for reuse"""
    global SENTENCE_TRANSFORMER
    if SENTENCE_TRANSFORMER is None:
        initialize_models()
    return SENTENCE_TRANSFORMER.encode([sentence])[0]

def initialize_models():
    """Initialize all models with proper device settings"""
    global BERT_SCORER, SENTENCE_TRANSFORMER, ABSTRACTIVE_PIPELINE
    
    try:
        # Initialize BERTScorer
        BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
        
        # Initialize Sentence Transformer
        SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')
        SENTENCE_TRANSFORMER = SENTENCE_TRANSFORMER.to(device)
        
        # Initialize T5 Pipeline with GPU support
        ABSTRACTIVE_PIPELINE = pipeline("summarization", 
                                      model="t5-small", 
                                      tokenizer="t5-small",
                                      device=0 if device == "cuda" else -1,
                                      batch_size=8)  # Added batch processing
        
        print("All models initialized successfully on", device)
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

# Initialize models at module level
initialize_models()

# 1. Φόρτωση δεδομένων
def load_data(filepath, num_samples=300):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Content", "Summary"])
    return df.sample(n=num_samples, random_state=42).reset_index(drop=True)

# 2. Preprocessing
def preprocess(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_target_length(reference_summary, mode='sentences'):
    """Calculate target length based on reference summary"""
    if mode == 'sentences':
        return len(sent_tokenize(reference_summary))
    else:  # words
        return len(reference_summary.split())

# 3. TF-IDF Summarization (extractive)
def tfidf_summarizer(text, reference_summary):
    target_sentences = max(1, get_target_length(reference_summary))
    sentences = sent_tokenize(text)
    if len(sentences) <= target_sentences:
        return text

    tfidf = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf, tfidf)
    doc_vector = np.asarray(tfidf.mean(axis=0))
    scores = np.asarray(cosine_similarity(tfidf, doc_vector)).flatten()
    top_indices = np.argsort(scores)[-target_sentences:]
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    return ' '.join(top_sentences)

# 4. TextRank with Sentence Embeddings
def textrank_summarizer(text, reference_summary):
    target_sentences = max(1, get_target_length(reference_summary))
    sentences = sent_tokenize(text)
    if len(sentences) <= target_sentences:
        return text

    # Process sentences in batches for better GPU utilization
    batch_size = 32
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = SENTENCE_TRANSFORMER.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    
    # Add small epsilon to avoid zero similarities
    similarity_matrix = similarity_matrix + 1e-6
    # Normalize the matrix
    row_sums = similarity_matrix.sum(axis=1)
    similarity_matrix = similarity_matrix / row_sums[:, np.newaxis]
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    try:
        scores = nx.pagerank(nx_graph, max_iter=200, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        scores = nx.degree_centrality(nx_graph)
    
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [s for _, s in ranked[:target_sentences]]
    return ' '.join(top_sentences)

# 5. Abstractive Summarization using T5
def abstractive_summarizer(text, reference_summary):
    target_length = len(reference_summary.split())  # Use word count for abstractive
    max_length = min(150, int(target_length * 1.2))  # Allow some flexibility
    min_length = max(10, int(target_length * 0.8))

    # Split long text into chunks of 500 characters with overlap
    max_chunk_length = 500
    chunks = []
    for i in range(0, len(text), max_chunk_length//2):
        chunk = text[i:i + max_chunk_length]
        if len(chunk) > 50:  # Only process chunks that are long enough
            chunks.append(chunk)
    
    if not chunks:
        return text
    
    # Process chunks in batches for better GPU utilization
    summaries = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        input_texts = ["summarize: " + chunk.strip() for chunk in batch]
        try:
            results = ABSTRACTIVE_PIPELINE(input_texts, 
                                         max_length=max_length,
                                         min_length=min_length,
                                         do_sample=False,
                                         truncation=True)
            summaries.extend(result['summary_text'] for result in results)
        except Exception as e:
            print(f"Warning: Error in abstractive summarization: {e}")
            continue
    
    # Combine summaries
    final_summary = ' '.join(summaries)
    return final_summary if final_summary.strip() else text

# 6. Αξιολόγηση με ROUGE και BERTScore
def evaluate_summary(reference, generated):
    scores = {}
    
    # ROUGE evaluation
    rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_inst.score(reference, generated)
    scores['rouge1_f1'] = rouge_scores['rouge1'].fmeasure
    scores['rouge2_f1'] = rouge_scores['rouge2'].fmeasure
    scores['rougeL_f1'] = rouge_scores['rougeL'].fmeasure
    
    # BERTScore evaluation
    try:
        P, R, F1 = BERT_SCORER.score([generated], [reference])
        scores['bertscore_f1'] = F1.mean().item()
    except Exception as e:
        print(f"Warning: Error in BERTScore calculation: {e}")
        scores['bertscore_f1'] = 0.0
    
    # Semantic Similarity using Sentence Transformer
    try:
        ref_embedding = SENTENCE_TRANSFORMER.encode(reference, convert_to_tensor=True)
        gen_embedding = SENTENCE_TRANSFORMER.encode(generated, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(ref_embedding, gen_embedding)
        scores['semantic_similarity'] = similarity.item()
    except Exception as e:
        print(f"Warning: Error in semantic similarity calculation: {e}")
        scores['semantic_similarity'] = 0.0
    
    # BLEU Score
    try:
        reference_tokens = [word_tokenize(reference)]
        generated_tokens = word_tokenize(generated)
        bleu_score = sentence_bleu(reference_tokens, generated_tokens)
        scores['bleu_score'] = bleu_score
    except Exception as e:
        print(f"Warning: Error in BLEU score calculation: {e}")
        scores['bleu_score'] = 0.0
    
    # Length Ratio
    try:
        ref_length = len(reference.split())
        gen_length = len(generated.split())
        length_ratio = gen_length / ref_length if ref_length > 0 else 0
        scores['length_ratio'] = length_ratio
    except Exception as e:
        print(f"Warning: Error in length ratio calculation: {e}")
        scores['length_ratio'] = 0.0
    
    return scores

# 8. Hybrid T5 & TextRank Summarization
def hybrid_summarizer(text, reference_summary):
    # First, get extractive summary using TextRank
    textrank_summary = textrank_summarizer(text, reference_summary)
    
    # Then, refine it using T5
    final_summary = abstractive_summarizer(textrank_summary, reference_summary)
    
    return final_summary

# 7. Main εκτέλεση
if __name__ == "__main__":
    import sys
    
    # Default to TFIDF if no method specified
    valid_methods = ["tfidf", "textrank", "t5", "hybrid", "bertscore", "all"]
    if len(sys.argv) > 1 and sys.argv[1].lower() in valid_methods:
        method = sys.argv[1].lower()
    else:
        print("\nAvailable methods:")
        print("1. tfidf - TF-IDF based summarization")
        print("2. textrank - TextRank based summarization")
        print("3. t5 - T5 based abstractive summarization")
        print("4. hybrid - Hybrid T5 & TextRank")
        print("5. bertscore - BERTScore based summarization")
        print("6. all - Run all methods")
        print("\nUsage: python summarization_methods.py [method]")
        sys.exit(1)

    print("Loading dataset (processing 300 articles)...")
    df = load_data("data.csv", num_samples=300)
    print(f"Loaded {len(df)} articles successfully.")

    results = {}
    last_article_summaries = {}
    
    print("\nGenerating summaries...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        article = preprocess(row['Content'])
        reference_summary = preprocess(row['Summary'])

        # Store the last article and its reference summary
        if idx == len(df) - 1:
            last_article_summaries['article'] = article
            last_article_summaries['reference'] = reference_summary

        if method == "tfidf" or method == "all":
            summary = tfidf_summarizer(article, reference_summary)
            scores = evaluate_summary(reference_summary, summary)
            if "TFIDF" not in results:
                results["TFIDF"] = []
            results["TFIDF"].append(scores)
            if idx == len(df) - 1:
                last_article_summaries['TFIDF'] = summary

        if method == "textrank" or method == "all":
            summary = textrank_summarizer(article, reference_summary)
            scores = evaluate_summary(reference_summary, summary)
            if "TextRank" not in results:
                results["TextRank"] = []
            results["TextRank"].append(scores)
            if idx == len(df) - 1:
                last_article_summaries['TextRank'] = summary

        if method == "t5" or method == "all":
            summary = abstractive_summarizer(article, reference_summary)
            scores = evaluate_summary(reference_summary, summary)
            if "T5" not in results:
                results["T5"] = []
            results["T5"].append(scores)
            if idx == len(df) - 1:
                last_article_summaries['T5'] = summary

        if method == "hybrid" or method == "all":
            summary = hybrid_summarizer(article, reference_summary)
            scores = evaluate_summary(reference_summary, summary)
            if "Hybrid" not in results:
                results["Hybrid"] = []
            results["Hybrid"].append(scores)
            if idx == len(df) - 1:
                last_article_summaries['Hybrid'] = summary

        if method == "bertscore" or method == "all":
            sentences = sent_tokenize(article)
            if len(sentences) > 1:
                # Calculate BERTScore for each sentence against the whole article
                _, _, F1 = BERT_SCORER.score(sentences, [article] * len(sentences))
                target_sentences = max(1, get_target_length(reference_summary))
                top_indices = torch.argsort(F1, descending=True)[:target_sentences]
                summary = ' '.join([sentences[i] for i in sorted(top_indices)])
            else:
                summary = article
            
            scores = evaluate_summary(reference_summary, summary)
            if "BERTScore" not in results:
                results["BERTScore"] = []
            results["BERTScore"].append(scores)
            if idx == len(df) - 1:
                last_article_summaries['BERTScore'] = summary

    # Print last article details
    print("\n" + "="*80)
    print("LAST ARTICLE DETAILS")
    print("="*80)
    
    print("\nORIGINAL ARTICLE:")
    print("-"*50)
    print(f"{last_article_summaries['article'][:500]}...")
    
    print("\nREFERENCE SUMMARY:")
    print("-"*50)
    print(last_article_summaries['reference'])
    
    print("\nGENERATED SUMMARIES:")
    print("-"*50)
    for method_name in last_article_summaries:
        if method_name not in ['article', 'reference']:
            if method == "all" or method == method_name.lower():
                print(f"\n{method_name} Summary:")
                print(f"{last_article_summaries[method_name]}")
                
                # Calculate and print ROUGE scores for this summary
                scores = evaluate_summary(last_article_summaries['reference'], 
                                       last_article_summaries[method_name])
                print("\nROUGE Scores:")
                print(f"ROUGE-1: {scores['rouge1_f1']:.4f}")
                print(f"ROUGE-2: {scores['rouge2_f1']:.4f}")
                print(f"ROUGE-L: {scores['rougeL_f1']:.4f}")
                print(f"BERTScore: {scores['bertscore_f1']:.4f}")

    # Print overall results
    print("\n" + "="*80)
    print("OVERALL EVALUATION RESULTS")
    print("="*80)
    for method_name in results:
        scores_df = pd.DataFrame(results[method_name])
        mean_scores = scores_df.mean()
        std_scores = scores_df.std()
        print(f"\n=== {method_name} ===")
        for metric in mean_scores.index:
            print(f"{metric}:")
            print(f"  Mean: {mean_scores[metric]:.4f}")
            print(f"  Std:  {std_scores[metric]:.4f}")
