import json
import pickle
from pathlib import Path

SRC = Path("/home/usama/Projects/PathomicFusion/pathomic/lib/python3.12/site-packages/geneformer/token_dictionary_gc104M.pkl")
DST = Path("data/TCGA_GBMLGG/geneformer/gene_token_map.json")

def main():
    if not SRC.exists():
        print(f"Source file not found: {SRC}")
        return

    with open(SRC, "rb") as f:
        token_map = pickle.load(f)

    token_map = {k: int(v) for k, v in token_map.items() if isinstance(k, str)}

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        json.dump(token_map, f)

    print(f"Saved token map to: {DST}")
    print(f"Number of genes in token map: {len(token_map)}")
    print("Sample entries:", list(token_map.items())[:10])

if __name__ == "__main__":
    main()