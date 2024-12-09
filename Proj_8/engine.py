import chromadb


class Embed(chromadb.EmbeddingFunction):
    def __init__(self, max_seq_length=2048):
        self.max_seq_length = max_seq_length
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-2k-retrieval", trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=max_seq_length
        )

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # Straight from the hugging face guide
        input_ids = self.tokenizer(
            input,
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        outputs = self.model(**input_ids)
        embeddings = outputs["sentence_embedding"].detach().cpu().numpy()
        return embeddings


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    SEQ_LEN = 2048
    

    persist_directory = "/data_out/Proj_8/db"

    dataset = load_dataset("hazyresearch/LoCoV1-Documents")["test"]
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    EXPECTED_LEN = dataset.shape[0]

    Embedder = Embed()
    db = chroma_client.create_collection(
        name="search_corpus", get_or_create=True, embedding_function=Embedder)

    if(db.count() < EXPECTED_LEN):
        print("Embedding documents now")
        print("Need to embed %d" % (EXPECTED_LEN - db.count()))
        batch_size = 1
        for i in range(db.count(),EXPECTED_LEN,batch_size):
            batch = dataset[i:i+batch_size]
            db.add(ids=batch['pid'],documents=batch['passage'])
            if i %5 == 0:
                print("Embedding document %d", i)

    else:
        print("Database Already Loaded")
        ##Means the model hasn't been filled yet

    print("Ready for Inference:")
    # search_term = input("Search Query: ")
    # results = db.query(query_texts=search_term)
    # print(results['ids'])


