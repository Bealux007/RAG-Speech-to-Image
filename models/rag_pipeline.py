from transformers import RagRetriever, AutoTokenizer, AutoModel
from datasets import load_dataset
import torch

class RAGPipeline:
    def __init__(self):
        dataset_path = "C:/Users/bharw/Documents/GitHub/vettura-genai/Assignments/Assignment_2.3/dataset"
        index_path = "C:/Users/bharw/Documents/GitHub/vettura-genai/Assignments/Assignment_2.3/index"
        
        # Load the dataset
        dataset = load_dataset('json', data_files='data/job_descriptions.json')
        
        # Compute embeddings
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        def compute_embeddings(batch):
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            batch['embeddings'] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            return batch
        
        # Process the dataset in smaller batches to avoid memory issues
        dataset = dataset.map(compute_embeddings, batched=True, batch_size=2)
        
        # Add FAISS index
        dataset['train'].add_faiss_index(column='embeddings')
        
        # Drop indexes before saving the dataset
        dataset['train'].drop_index('embeddings')
        
        # Save dataset to disk
        dataset.save_to_disk(dataset_path)
        
        # Reload the dataset and add FAISS index
        dataset = load_dataset(dataset_path)
        dataset['train'].add_faiss_index(column='embeddings')
        
        # Save the FAISS index to disk
        dataset['train'].get_index('embeddings').save(index_path)
        
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", 
            index_name="custom", 
            passages_path="data/job_descriptions.json",
            dataset_path=dataset_path,
            index_path=index_path
        )