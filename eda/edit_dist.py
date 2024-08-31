# %%
from datasets import load_dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool

CPU_COUNT = 12
N_SAMPLES = 30
TASKS = [
    "beta_lactamase_complete",
    "beta_lactamase_unique",
    "stability",
    # "melting_point", --> this one breaks bc of max recusion depth
    "fluorescence"
]

class Solution:
    def __init__(self):
        self.cache = {}
    
    def minDistance(self, word1: str, word2: str) -> int:
        def dfs(i, j):
            if (i,j) in self.cache: return self.cache[(i,j)]
            if i == len(word1) and j == len(word2): return 0
            if i == len(word1): return len(word2) - j
            if j == len(word2): return len(word1) - i
            
            if word1[i] == word2[j]: self.cache[(i,j)] = dfs(i+1, j+1)
            else:
                cand1 = 1 + dfs(i, j+1) # delete a ch from word2
                cand2 = 1 + dfs(i+1, j) # delete a ch from word1
                cand3 = 1 + dfs(i+1, j+1) # replace or insert to match
                self.cache[(i,j)] = min(cand1, cand2, cand3)
                
            return self.cache[(i,j)]

        return dfs(0,0)

def compute_distance(pair):
    """For Pool, we need to wrap it in a fn"""
    i, j, seqs = pair
    S = Solution()
    return S.minDistance(seqs[i], seqs[j])
# %%
for TASK in TASKS:
    dataset = load_dataset(
        "InstaDeepAI/true-cds-protein-tasks", name=TASK, trust_remote_code=True
    )

    dataloader = DataLoader(dataset["train"], batch_size=1, shuffle=False)

    seqs = []
    for i, batch in enumerate(dataloader):
        if i >= N_SAMPLES: break
        seqs.append(batch['sequence'][0])
    
    pairs = [(i, j, seqs) for i in range(N_SAMPLES) for j in range(i+1, N_SAMPLES)]
    # print(len(pairs))
    with Pool(CPU_COUNT) as pool:
        dists = pool.map(compute_distance, pairs)

    print("Mean edit dist for {}: {}".format(TASK, sum(dists) / len(dists)))
