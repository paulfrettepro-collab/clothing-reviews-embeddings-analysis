
# 🧠 Topic Analysis of Clothing Reviews with Embeddings

## 🎯 Objectif du projet

Ce projet vise à analyser les retours clients d’un site e-commerce de vêtements afin de comprendre les thématiques principales et les sentiments exprimés dans les avis.  
L’objectif est d’utiliser la puissance des embeddings et de la similarité cosinus pour :

* Identifier automatiquement les sujets récurrents (qualité, coupe, style, confort…)
* Visualiser les relations entre les avis dans un espace vectoriel 2D
* Trouver les avis les plus similaires pour améliorer les réponses clients et la personnalisation du service

---

## 🧩 Données utilisées

**Dataset :** `womens_clothing_e-commerce_reviews.csv`  
Ce jeu de données contient des avis clients sur des articles de mode féminine vendus en ligne.

| Colonne       | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `Review Text` | Commentaire client sur le produit (qualité, taille, style, confort...) |

Chaque avis textuel a été transformé en vecteur numérique à l’aide d’un modèle d’embedding d’OpenAI (`text-embedding-3-small`).

---

## 🧱 Étapes principales du projet

### 1. Génération des embeddings

Chaque review est encodée en vecteur de 1 536 dimensions grâce au modèle OpenAI Embeddings API.  
Cela permet de représenter le sens de chaque phrase de façon mathématique.

```
from openai import OpenAI
client = OpenAI()
embeddings = [d.embedding for d in client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
).data]
```

Variable finale :

```
embeddings  # list of lists
```

---

### 2. Réduction de dimension (t-SNE)

Pour visualiser les relations entre les avis dans un espace 2D, on applique t-SNE :

```
from sklearn.manifold import TSNE

embeddings_2d = TSNE(
    n_components=2,
    learning_rate="auto",
    init="pca",
    perplexity=30
).fit_transform(emb_matrix)
```

Chaque point représente un avis, et les clusters reflètent des thématiques communes.

---

### 3. Catégorisation automatique des topics

On compare les embeddings de chaque avis à ceux de mots-clés thématiques (`quality`, `fit`, `style`, `comfort`, etc.).  
Le topic le plus similaire est attribué à chaque review :

```
from sklearn.metrics.pairwise import cosine_similarity

topics = ["quality", "fit", "style", "comfort", "color", "material"]
topic_embeddings = get_openai_embeddings(topics)

sims = cosine_similarity(emb_matrix, np.array(topic_embeddings))
top_topic_idx = sims.argmax(axis=1)
topic_df = pd.DataFrame({
    "review": texts,
    "top_topic": [topics[i] for i in top_topic_idx]
})
```

Résultat : un DataFrame indiquant le thème dominant de chaque avis.

---

### 4. Recherche d’avis similaires

On construit une fonction pour trouver les 3 avis les plus proches d’un texte donné, selon la similarité cosinus :

```
def most_similar_reviews_fn(query_text, doc_texts, doc_embs, k=3):
    q_emb = get_openai_embeddings([query_text])
    sim = cosine_similarity([q_emb], np.array(doc_embs))
    idxs = np.argsort(sim)[::-1][:k]
    return [doc_texts[i] for i in idxs]
```

Exemple :

```
query = "Absolutely wonderful - silky and sexy and comfortable"
most_similar_reviews = most_similar_reviews_fn(query, texts, embeddings)
```

Résultat : une liste de 3 avis exprimant des sentiments similaires.

---

## 📊 Visualisation

Une représentation 2D des embeddings met en évidence les regroupements d’avis selon leurs similarités sémantiques :

```
plt.figure(figsize=(6,5))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=15)
plt.title("2D visualization of clothing reviews")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
```

Chaque cluster représente une thématique dominante (confort, coupe, qualité…).

---

## 🧠 Résultats clés

* Les embeddings permettent d’identifier les thèmes cachés dans les avis sans supervision.
* Le modèle capture des nuances comme :
  * “Too small, uncomfortable” → “fit” / “comfort”
  * “Beautiful material and design” → “style” / “quality”
* L’approche de similarité vectorielle offre une base solide pour :
  * Des recommandations personnalisées
  * Une analyse de sentiment automatisée
  * Un moteur de recherche sémantique

---

## 🧰 Stack technique

| Catégorie         | Outils utilisés                        |
| ----------------- | -------------------------------------- |
| Langage           | Python                                 |
| API               | OpenAI Embeddings API                  |
| Machine Learning  | Scikit-learn (TSNE, cosine similarity) |
| Visualisation     | Matplotlib                             |
| Data Manipulation | Pandas, NumPy                          |
| Notebook          | Datacamp Workspace / Jupyter           |

---

## 💡 Ce que ce projet démontre

* La compréhension des embeddings et de leur usage pour représenter le sens d’un texte.
* La maîtrise de la similarité cosinus pour mesurer la proximité sémantique.
* La capacité à structurer une analyse de texte de bout en bout, depuis la préparation jusqu’à l’interprétation visuelle.
* Une logique réutilisable pour tout projet de moteur de recherche sémantique, analyse d’opinions, ou recommandation intelligente.

---

## 🚀 Pistes d’amélioration

* Utiliser ChromaDB ou FAISS pour accélérer la recherche de similarité.
* Ajouter une analyse de sentiment supervisée (positif / neutre / négatif).
* Construire un dashboard Streamlit pour l’exploration interactive des clusters.
* Tester d’autres modèles d’embeddings (ex : `text-embedding-3-large`, `all-MiniLM-L6-v2` de Sentence Transformers).

---

## 👨‍💻 Auteur

Paul Fretté  
Data Analyst / Data Engineer Freelance  
🌐 [GitHub](https://github.com/paulfrettepro-collab) • [LinkedIn](https://linkedin.com/in/paulfrette) • [Bento](https://bento.me/paulfrette)
