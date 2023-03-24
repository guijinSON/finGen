import torch 
from torch.utils.data import Dataset, DataLoader
       
class Seq2SeqDataset(Dataset):
    def __init__(self, src, tgt, tokenizer):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]
        seq2seq = src + self.tokenizer.eos_token + tgt

        return {
            'seq2seq':seq2seq
        }


class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer
                 ):
        
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        seq = [item['seq2seq'] for item in batch]
        seq_tokenized = self.tokenize(seq)
        seq_attenton_mask = self.update_mask(seq_tokenized.input_ids, seq_tokenized.attention_mask)

        return {
            'seq_input_ids': seq_tokenized.input_ids, 
            'seq_attention_mask': seq_tokenized.attention_mask,
            }

    def tokenize(self,input_str):
        return  self.tokenizer.batch_encode_plus(input_str, 
                                                    padding='longest', 
                                                    max_length=512,
                                                    truncation=True, 
                                                    return_tensors='pt')
        
    def update_mask(self, seq_input_ids, seq_attention_mask):
        eos_idx = (seq_input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
        for dim1,dim2 in eos_idx:
            seq_attention_mask[dim1,:dim2] = 0  
        return seq_attention_mask

def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=batch_generator,
                              drop_last=True,
                              num_workers=4)
    return data_loader
