# simple script for embedding papers using huggingface Specter
# requirement: pip install --upgrade transformers==4.2.2
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch

class Dataset:

    def __init__(self, data_path, max_length=512, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.max_length = max_length
        self.batch_size = batch_size
        # data is assumed to be a json file
        with open(data_path) as f:
            raw_data = json.load(f)
            # Nếu data là list (như papers_clean.json), chuyển về dạng dict để ổn định logic
            if isinstance(raw_data, list):
                self.data = {item['paper_id']: item for item in raw_data}
            else:
                self.data = raw_data

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        
        # Chạy trên CPU hay GPU (Kaggle hỗ trợ CUDA)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i, (k, d) in enumerate(self.data.items()):
            batch_ids.append(k)
            batch.append(d['title'] + ' ' + (d.get('abstract') or ''))
            
            if (i + 1) % batch_size == 0:
                input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                           return_tensors="pt", max_length=self.max_length)
                yield input_ids.to(device), batch_ids
                batch_ids = []
                batch = []
        
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=self.max_length)        
            yield input_ids.to(device), batch_ids

class Model:

    def __init__(self):
        self.model = AutoModel.from_pretrained('allenai/specter')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :] # cls token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                        'the output format is jsonlines where each line has "paper_id" and "embedding" keys')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')

    args = parser.parse_args()
    dataset = Dataset(data_path=args.data_path, batch_size=args.batch_size)
    model = Model()
    results = {}
    batches = []
    for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
        batches.append(batch)
        emb = model(batch)
        for paper_id, embedding in zip(batch_ids, emb.unbind()):
            results[paper_id] =  {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
