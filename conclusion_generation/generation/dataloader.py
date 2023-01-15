import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.sentence = self.data.sentence

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        sentence = str(self.sentence[index])
        sentence = " ".join(sentence.split())

        source = self.tokenizer.batch_encode_plus(
            [sentence],
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
        }
