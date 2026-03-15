from PIL import Image
from io import BytesIO
import base64
import re
import torch
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX, TSF_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_TSF_TOKEN


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    tsf_token_index=TSF_TOKEN_INDEX,
    return_tensors=None,
):
    pattern = rf"({re.escape(DEFAULT_IMAGE_TOKEN)}|{re.escape(DEFAULT_TSF_TOKEN)})"
    chunks = re.split(pattern, prompt)

    input_ids = []
    first_text_chunk = True
    for chunk in chunks:
        if chunk == "":
            continue
        if chunk == DEFAULT_IMAGE_TOKEN:
            input_ids.append(image_token_index)
            continue
        if chunk == DEFAULT_TSF_TOKEN:
            input_ids.append(tsf_token_index)
            continue

        chunk_ids = tokenizer(chunk).input_ids
        if first_text_chunk:
            input_ids.extend(chunk_ids)
            first_text_chunk = False
        else:
            # Avoid duplicating BOS for non-leading chunks.
            if len(chunk_ids) > 0 and chunk_ids[0] == tokenizer.bos_token_id:
                chunk_ids = chunk_ids[1:]
            input_ids.extend(chunk_ids)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
