import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from .prompt_templates import (
    Caption_templates,
    pelvis_ct_prompts,
    chest_ct_prompts,
    abdomen_ct_prompts,
)
import nibabel as nib
from functools import partial


class AmosCapDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 128), mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        df = pd.read_csv(
            args.amos_train_cap_data_path
            if mode == "train"
            else args.amos_validation_cap_data_path
        )
        self.images_path = df["image"]
        self.captions = df["caption"]
        self.organs = df["label"]

        self.nii_to_tensor = partial(
            self.__nii_img_to_tensor, target_shape=target_shape
        )

    def __nii_img_to_tensor(self, path, target_shape):
        img_data = nib.load(path)
        img_data = img_data.get_fdata()
        img_data = img_data.astype(np.float32)

        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = img_data * 1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data + 400) / 600)).astype(np.float32)
        slices = []
        # Use this part only for m3d
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        tensor = torch.tensor(img_data)

        # Get the dimensions of the input tensor

        # Extract dimensions
        h, w, d = tensor.shape
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=0,
        )

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                image = self.nii_to_tensor(
                    os.path.join(self.data_root, self.images_path[idx])
                )

                answer = self.captions[idx]

                if self.organs[idx] in "pelvis":
                    prompt_question = random.choice(pelvis_ct_prompts)
                elif self.organs[idx] in "abdomen":
                    prompt_question = random.choice(abdomen_ct_prompts)
                elif self.organs[idx] in "chest":
                    prompt_question = random.choice(chest_ct_prompts)
                else:
                    print(
                        f"Error in __getitem__: Unrecognized organ!=={self.organs[idx]}======"
                    )

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )  # <IMG_TOKENS><QUESTION>' '<ANSWER/CAP>

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": "",
                    "question_type": "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.images_path) - 1)
