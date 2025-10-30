"""
Snippets utiles pour réutiliser le projet.
Chaque fonction est autonome. Copie-colle selon le besoin.
"""

from typing import List
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


EMB_MODEL = "text-embedding-3-small"


def embed_texts(texts: List[str], client: OpenAI) -> List[List[float]]:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def query_similar(query: str, docs: List[str], doc_embs_ll: List[List[float]], client: OpenAI, k: int = 3) -> List[str]:
    q = np.array(embed_texts([query], client)[0]).reshape(1, -1)
    mat = np.array(doc_embs_ll)
    sims = cosine_similarity(q, mat)[0]
    idxs = np.argsort(sims)[::-1][:k]
    return [docs[i] for i in idxs]


def label_with_topics(doc_embs_ll: List[List[float]], topics: List[str], client: OpenAI) -> pd.DataFrame:
    doc_mat = np.array(doc_embs_ll)
    topic_mat = np.array(embed_texts(topics, client))
    sims = cosine_similarity(doc_mat, topic_mat)
    top_idx = sims.argmax(axis=1)
    return pd.DataFrame({"top_topic": [topics[i] for i in top_idx]})


def tsne_2d(doc_embs_ll: List[List[float]]) -> np.ndarray:
    mat = np.array(doc_embs_ll)
    perplexity = int(max(5, min(30, len(mat) // 3)))
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=perplexity)
    return tsne.fit_transform(mat)


def plot_points(points_2d: np.ndarray, outpath: str = "plot_2d.png") -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=12)
    plt.title("2D visualization (t-SNE)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def demo_minimal(texts: List[str]) -> None:
    client = OpenAI()
    embs = embed_texts(texts, client)
    pts2d = tsne_2d(embs)
    plot_points(pts2d)
    print("Done. Files: plot_2d.png")
