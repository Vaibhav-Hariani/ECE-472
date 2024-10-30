import tensorflow as tf
from adam import Adam
from datasets import load_dataset
from MLP import MLP
from sentence_transformers import SentenceTransformer
from utils import restructure

if __name__ == "__main__":
    import math
    import os

    import numpy as np
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    SAVE_PATH = os.path.join("Proj_5", "MLP")

    ##Just get training data
    ds = load_dataset("fancyzhx/ag_news", split="train")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    BATCH_SIZE = 100
    NUM_ITERS = 500

    VALIDATE_SPLIT = 0.95

    bar = trange(NUM_ITERS)

    model = MLP(
        num_inputs=384,
        num_outputs=4,
        num_hidden_layers=7,
        hidden_layer_width=256,
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.softmax,
        dropout_rate=0.25,
    )

    optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)

    n_min = 0.1
    n_max = 2
    epochs = 0
    size = int(len(ds["label"]) * VALIDATE_SPLIT)
    total_epochs = BATCH_SIZE * NUM_ITERS / size

    validation_slice = np.arange(size, len(ds["label"]))
    validation_ds = ds.select(validation_slice)
    # embeddings = embedder.encode(ds['text'])
    val_embeddings = embedder.encode(validation_ds["text"])
    # val_embeddings = embeddings[size:]
    accuracy = 0

    for i in bar:
        batch_indices = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        sample = ds.select(batch_indices)
        embeddings = embedder.encode(sample["text"])
        # sample_embeddings = embeddings[batch_indices]
        expected = restructure(sample["label"], BATCH_SIZE, 4)
        with tf.GradientTape() as tape:
            predicted = model(embeddings, True)
            loss = tf.keras.losses.categorical_crossentropy(expected, predicted)
            loss = tf.math.reduce_mean(loss)
        ##Cosine annealing
        n_t = (
            n_min
            + (n_max - n_min) * (1 + tf.math.cos(epochs * math.pi / total_epochs)) / 2
        )
        epochs += BATCH_SIZE / size
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.train(grads=grads, vars=model.trainable_variables, decay_scale=n_t)
        if i % 10 == 9:
            if i % 100 == 99:
                expected = validation_ds["label"]
                model_out = model(val_embeddings, False)
                model_out = np.argmax(model_out, axis=1)
                accuracy = np.sum(model_out == expected) / validation_ds.num_rows
            bar.set_description(
                f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}, Accuracy => {accuracy:0.3f}:"
            )
            bar.refresh()

    tf.saved_model.save(model, SAVE_PATH)

    train_set = load_dataset("fancyzhx/ag_news", split="test")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = embedder.encode(train_set["text"])
    expected = train_set["label"]
    model_output = model(embeddings, False)
    model_output = np.argmax(model_output, axis=1)
    accuracy = np.sum(model_output == expected) / train_set.num_rows
    print("On test set, achieved accuracy of %0.1f%%" % (100 * accuracy))
