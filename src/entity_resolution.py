import pandas as pd
from rapidfuzz import fuzz, process
from typing import List, Tuple
from .config import THRESHOLDS

def blocking_key(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    parts = name.lower().split()
    if not parts:
        return ""
    key = "".join([p[:2] for p in parts])  # first 2 chars of tokens
    return key

def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: idx_a, idx_b, score, cluster_id
    Simple blocking + name similarity; can be extended with phone/email.
    """
    work = df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
    if "full_name_clean" not in work.columns:
        work["full_name_clean"] = work.get("full_name", "").astype(str).str.strip()
    work["block"] = work["full_name_clean"].apply(blocking_key)
    blocks = work.groupby("block")

    pairs: List[Tuple[int,int,float]] = []
    for blk, g in blocks:
        if blk == "" or len(g) < 2:
            continue
        names = list(g["full_name_clean"])
        ids = list(g["row_id"])
        # Compare within block
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = fuzz.token_sort_ratio(names[i], names[j])
                if score >= THRESHOLDS.name_similarity_min:
                    pairs.append((ids[i], ids[j], score))

    dup_df = pd.DataFrame(pairs, columns=["idx_a", "idx_b", "score"])
    if dup_df.empty:
        dup_df["cluster_id"] = []
        return dup_df

    # Build clusters via union-find-ish connectivity
    from collections import defaultdict, deque
    graph = defaultdict(set)
    for _, r in dup_df.iterrows():
        graph[r["idx_a"]].add(r["idx_b"])
        graph[r["idx_b"]].add(r["idx_a"])

    visited = set()
    cluster_id = 0
    node_to_cluster = {}
    for node in graph:
        if node in visited:
            continue
        cluster_nodes = []
        q = deque([node])
        visited.add(node)
        while q:
            u = q.popleft()
            cluster_nodes.append(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        for n in cluster_nodes:
            node_to_cluster[n] = cluster_id
        cluster_id += 1

    def assign_cluster(row):
        a = row["idx_a"]; b = row["idx_b"]
        return node_to_cluster.get(a, node_to_cluster.get(b, -1))
    dup_df["cluster_id"] = dup_df.apply(assign_cluster, axis=1)
    return dup_df