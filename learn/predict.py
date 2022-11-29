import torch.nn.functional as F

import torch

import numpy as np

from torch.autograd import Variable



def preprocess(text, dicts, max_length):
    w2ind = dicts["w2ind"]

    text = [
        int(w2ind[w] + 1) if w in w2ind else len(w2ind) + 1 for w in text.split()
    ]  # OOV words are given a unique index at end of vocab lookup
    # truncate long documents
    if len(text) > max_length:
        text = text[:max_length]
    else:
        text.extend([0] * (max_length - len(text)))

    data = np.array(text)[np.newaxis, ...]
    # assert False
    
    return  data

def get_preds(text, k, model, dicts, max_length=2500, gpu=False):
    data = preprocess(text, dicts, max_length)
    print("WTF")
    data = Variable(torch.LongTensor(data), volatile=True)
    if gpu:
        data = data.cuda()
    model.zero_grad()
    output, *_ = model(data)

    return postprocess(output, k, dicts, text)


def postprocess(output, k, dicts, text):
    ind2c, label_desc = dicts["ind2c"], dicts['label_desc']
    output = F.sigmoid(output)
    output = output.data.cpu().numpy().squeeze()

    # top-k predictions:
    top_k_indices = np.argsort(output)[::-1][:k]

    return {
        "text": text,
        "result": [
            {
                "code": ind2c[i],
                "name": label_desc.get(ind2c[i], "Unknown description..."),
                "probability": float(output[i]),
            }
            for i in top_k_indices
        ],
    }


