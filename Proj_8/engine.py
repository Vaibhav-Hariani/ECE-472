import chromadb
import numpy as np


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
        # print("embedded something")
        return embeddings

def get_l2(source, embeddings):
    db_val = db.get([source],include=['embeddings'])['embeddings'][0]
    difference = (db_val - embeddings)**2
    sum = np.sum(difference)
    return np.sqrt(sum)

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import trange

    SEQ_LEN = 2048
    INFER = True

    persist_directory = "db_cos_retry"

    chroma_client = chromadb.PersistentClient(path=persist_directory)

    Embedder = Embed()
    db = chroma_client.create_collection(
        name="search_corpus",
        get_or_create=True,
        embedding_function=Embedder,
        metadata={"hnsw:space": "cosine"},
    )

    if db.count() == 0:
        print("Embedding documents now")
        dataset = load_dataset("hazyresearch/LoCoV1-Documents")["test"]
        EXPECTED_LEN = dataset.shape[0]
        print("Need to embed %d" % (EXPECTED_LEN - db.count()))
        bar = trange(EXPECTED_LEN)
        for i in bar:
            batch = dataset[i] 
            ##Document name insertion to improve relevant document searching, especially for wikimedia pages & the like
            ##Should be most effective for scripts and other documents that have implicit structure
            document = "Document Type: " + batch["dataset"] + "\n" +batch["passage"]           
            db.add(ids=batch["pid"], documents=document)
            if i % 5 == 0:
                # print("Embedding document %d" % (i))
                bar.refresh()

    else:
        print("Database Already Loaded")
        print("Documents already embedded %d" % (db.count()))
        ##Means the model hasn't been filled yet

    if INFER:
        print("Ready for Inference:")
        search_term = input("Search Query (***EXIT*** to quit): ")
        while search_term != "***EXIT***":
            search_term = search_term
            embeddings = Embedder(search_term)
            results = db.query(query_embeddings=embeddings,n_results=10)
            print(results["ids"][0])
            threshold = (results["distances"][0][0] * 3)/2
            print(results["distances"][0])
            ##Trim irrelevant documents by only searching within k of the query: should improve 
            # precision at the expense of recall for documents with high relation
            for ind in range(len(results["documents"][0])):
                if(results["distances"][0][ind] < threshold):
                    f = open("Passages/" + results["ids"][0][ind]+".txt", "w")
                    f.write(results["documents"][0][ind])
                    f.write("\n ***END OF DOCUMENT *** \n")
                    f.close()
                else:
                    print("Document was not relevant enough")
            search_term = input("Search Query (***EXIT*** to quit): ")
