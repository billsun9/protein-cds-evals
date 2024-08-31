<h2 style="text-align:center">Basic Pytorch Experiments on Instadeep's Protein Regression Datasets</h2>

<h3>Overview:</h3>

- Dataset: [Instadeep Protein Tasks Dataset](https://huggingface.co/datasets/InstaDeepAI/true-cds-protein-tasks)
- I have pretty good reason to believe that the results of [their ICLR workshop paper](https://openreview.net/pdf/67236cf1ef9603efe6490e115cbf2affb80447cf.pdf), either of their model and/or their released dataset have issues. Essentially, I am unable to replicate the results of their paper (which has fairly poor descriptions of how they set up their experiments).
- This is a set of experiments illustrating some of these issues - which I believe mostly stems from low variance in the input space of the DNA sequences (e.g. in "fluorescence" task, the sequences of length 700 have average pairwise edit distance of []). In turn, this results in pre-trained models embedding sequences in roughly the same space, making it difficult for downstream FFNs to regress for target values.

<h3>Experiments</h3>

- EDA
- Pairwise Edit distance between 100 random samples of each of the datasets
- Finetuning experiments
  - DNABERT2, Nucleotide Transformer
  - Various hyperparameter configurations (custom losses, LRs, FFN complexity)
  - Baseline of random uniform embeddings