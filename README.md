# Improving Neural Machine Translation (NMT) using Pre-Trained Embeddings: FR-EN Case Study

### Description
This study aims to conduct an analysis of Neural Machine Translation (NMT) and transformer architectures, with a specific focus on embeddings. The research involves a case study of the Multi30k dataset to investigate the French-English translation task. Starting from encoder and transformer baseline models, we further propose two additional architectures that incorporate, in the embedding step, pre-trained word representations from BERT and XLNet LLMs. 

### Repository structure
The directory contains all the files needed to run the full notebook which gathers the results of our analysis.
The notebook begins with the data preprocessing sections (data retrieval, tokenization, data loaders, train/evaluate/translate functions, etc.) followed by the 4 main sections. Each of the these sections has: a call to the specific model, its training, a print of some translations, and valuation perplexity and BLEU scores as evaluation metrics.
Following the notebook's order, we have:
1) Transformer baseline whose code is found in Transformer_baseline.py
2) Transformer improved following the paper "Attention is all you need" in Transformer.py
3) Transformer using BERT embeddings in Transformer_BERT.py
4) Transformer using XLNet embeddings in Transformer_XLNet.py 


Extra) Transformer using GPT2 embeddings in Transformer_GPT.py not actually reported but tested during the project.


