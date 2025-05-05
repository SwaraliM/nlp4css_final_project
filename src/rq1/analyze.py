import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


for model_name in ["bertopic", "lda", "top2vec"]:
    df = pd.read_csv(f"{model_name}_summary.csv")

    group_keywords = df.groupby("group")["keywords"].apply(lambda ks: [word.strip() for k in ks for word in k.split(",")])
    group_top_words = group_keywords.apply(lambda words: Counter(words).most_common(10))

    top_words_data = []
    for group, top_words in group_top_words.items():
        row = {"group": group}
        for i, (word, _) in enumerate(top_words):
            row[f"top_word_{i+1}"] = word
        top_words_data.append(row)

    top_words_df = pd.DataFrame(top_words_data)
    top_words_df.to_csv(f"{model_name}_top_words.csv", index=False)
    
    # # Plotting the coherence score per group
    # df.groupby("group")["coherence"].mean() \
    #   .sort_values(ascending=False) \
    #   .plot(kind="bar", title="Average Topic Coherence per Group")

    # plt.ylabel("Coherence Score")
    # plt.xlabel("Group")
    # plt.tight_layout()
    # plt.show()

    unique_group_coherence = df[["group", "coherence"]].drop_duplicates()
    unique_group_coherence.to_csv(f"{model_name}_group_coherence.csv", index=False)