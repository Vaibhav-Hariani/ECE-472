import tensorflow as tf
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    import os

    import numpy as np

    MODEL_PATH = os.path.join("Proj_5", "MLP")
    model = tf.saved_model.load(MODEL_PATH)
    ds = load_dataset("fancyzhx/ag_news", split="test")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(ds["text"])
    expected = ds["label"]

    model_output = model(embeddings, False)
    model_output = np.argmax(model_output, axis=1)
    accuracy = np.sum(model_output == expected) / expected.size
    print("On test set, achieved accuracy of %0.1f%%" % (100 * accuracy))
