class Embedder:
    def __init__(self, item_to_id, embeddings):
        assert len(item_to_id) == len(embeddings), 'Items and embeddings length mismatch'
        self.item_to_id = item_to_id
        self.embeddings = embeddings
