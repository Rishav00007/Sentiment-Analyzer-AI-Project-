# # # # dataset.py

# # # import pandas as pd
# # # import torch
# # # from torch.utils.data import Dataset

# # # class AmazonReviewDataset(Dataset):
# # #     def __init__(self, csv_file, tokenizer, src_seq_len):
# # #         # Load the CSV file directly without any label conversion
# # #         self.data = pd.read_csv(csv_file)
# # #         self.tokenizer = tokenizer
# # #         self.src_seq_len = src_seq_len
        
# # #         # Print some debug info
# # #         print(f"Loaded {len(self.data)} samples")
# # #         print("Label distribution:", self.data['label'].value_counts())

# # #     def __len__(self):
# # #         return len(self.data)

# # #     def __getitem__(self, idx):
# # #         review = str(self.data.iloc[idx]["review"])
# # #         # Get label directly from CSV without any conversion
# # #         label = int(self.data.iloc[idx]["label"])

# # #         # Tokenize and truncate/pad
# # #         tokens = self.tokenizer.encode(
# # #             review,
# # #             max_length=self.src_seq_len,
# # #             padding="max_length",
# # #             truncation=True
# # #         )

# # #         return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)



# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class AmazonReviewDataset(Dataset):
#     def __init__(self, csv_file, tokenizer, src_seq_len):
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = tokenizer
#         self.src_seq_len = src_seq_len
        
#         print(f"Loaded {len(self.data)} samples")
#         print("Label distribution:", self.data['label'].value_counts())

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         review = str(self.data.iloc[idx]["review"])
#         label = int(self.data.iloc[idx]["label"])

#         # Use tokenizer method (recommended)
#         tokens = self.tokenizer(
#             review,
#             max_length=self.src_seq_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"  # returns dict of tensors
#         )

#         # Return input_ids only
#         input_ids = tokens["input_ids"].squeeze(0)  # shape: [seq_len]

#         return input_ids, torch.tensor(label, dtype=torch.long)


import pandas as pd
import torch
from torch.utils.data import Dataset

class AmazonReviewDataset(Dataset):
    def __init__(self, csv_file, tokenizer, src_seq_len):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.src_seq_len = src_seq_len

        # Optional: Clean labels to integers if accidentally floats/strings
        self.data['label'] = pd.to_numeric(self.data['label'], errors='coerce').astype('Int64')

        # Keep only valid labels: 0, 1, 2
        self.data = self.data[self.data['label'].isin([0, 1, 2])].reset_index(drop=True)

        print(f"Loaded {len(self.data)} samples")
        print("Label distribution:", self.data['label'].value_counts())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = str(self.data.iloc[idx]["review"]).strip().lower()  # Clean text
        label = int(self.data.iloc[idx]["label"])

        # Tokenize
        tokens = self.tokenizer(
            review,
            max_length=self.src_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)  # shape: [seq_len]

        # Final label check
        assert label in [0, 1, 2], f"Invalid label: {label}"

        return input_ids, torch.tensor(label, dtype=torch.long)



# import pandas as pd 
# import torch
# from torch.utils.data import Dataset

# class AmazonReviewDataset(Dataset):
#     def __init__(self, csv_file, tokenizer, src_seq_len):
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = tokenizer
#         self.src_seq_len = src_seq_len
        
#         print(f"Loaded {len(self.data)} samples")
#         print("Label distribution:", self.data['label'].value_counts())

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         review = str(self.data.iloc[idx]["review"])
#         label = int(self.data.iloc[idx]["label"])

#         tokens = self.tokenizer(
#             review,
#             max_length=self.src_seq_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         input_ids = tokens["input_ids"].squeeze(0)  # [seq_len]

#         return input_ids, torch.tensor(label, dtype=torch.long)


