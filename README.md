# Earnings-Call-Analysis-System

Earning Call Transcripts:  

Inside the Transcript Folder  


Data prep:

DSA4265_01_data_prep.ipynb

BERT:

Data: labelled_sentences.xlsx

3 models:

Baseline: DSA4265_Project_BERT_baseline_without_pseudo_labelling_.ipynb  
Gradual Unfreezing: DSA4265_Project_BERT_gradual_unfreezing.ipynb  
Final model with filtering of pseudo-labeling: DSA4265_Project_BERT_final_version_with_filtering.ipynb


RAG:

DSA4265_RAG_V2.ipynb covers the chunking strategy (long-turn splitting + sliding window), FAISS indexing, baseline retrieval and evaluation metrics.

Final Pipeline:

Final_FinBERT_RAG_Pipeline.ipynb covers the multi-task model architecture (sentiment + topic heads), the ask_auto() end-to-end pipeline with Gemini, and the evaluation results table comparing all three retrieval strategies.
