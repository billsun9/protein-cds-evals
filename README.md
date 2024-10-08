<h2 style="text-align:center">Basic Pytorch Experiments on Instadeep's Protein Regression Datasets</h2>

<h3>Overview:</h3>

- Dataset: [Instadeep Protein Tasks Dataset](https://huggingface.co/datasets/InstaDeepAI/true-cds-protein-tasks)
- I have pretty good reason to believe that the results of [their ICLR workshop paper](https://openreview.net/pdf/67236cf1ef9603efe6490e115cbf2affb80447cf.pdf), either of their model and/or their released dataset have issues. Essentially, I am unable to replicate the results of their paper (which has fairly poor descriptions of how they set up their experiments).
- This is a set of experiments illustrating some of these issues - which I believe mostly stems from low variance in the input space of the DNA sequences (e.g. in "fluorescence" task, the sequences of length 700 have average pairwise edit distance of 5.24, a 99.4% similarity). In turn, this results in pre-trained models embedding sequences in roughly the same space, making it difficult for downstream FFNs to regress for target values.

<h3>Experiments</h3>

- EDA (see `eda/scalar_regression_eda.py` and `eda/ssp_eda.py`)
- Pairwise Edit distance between 100 random samples of each of the datasets (`eda/ssp.py`)
- Finetuning experiments
  - DNABERT2, Nucleotide Transformer
  - Various hyperparameter configurations (custom losses, LRs, FFN complexity)
  - Baseline of random uniform embeddings

<h3>Results</h3>

- No experiment using DNABERT2 or nucleotide transformer as a pretrained model is able to achieve results even close to what is described in the paper. The models consistently regress to the mean on all listed scalar regression datasets.
- Neither DNABERT2 nor nucleotide transformer appear to be able to overfit on a very small (n=100) sample of the training set. Instead, it still regresses to the mean.
- Observing custom defined distance metrics (see `utils.py`) between all 100 of the trained embeddings, it appears as if the pretrained models' embeddings get closer in space as training occurs, usually converging within 3 epochs
- The baseline of frozen random embeddings (torch.rand(size=(hidden_dim,)) in `random_embeddings.py`) with a FFN is able to have perfect training accuracy as expected. The distances of its embeddings obviously stays the same over training