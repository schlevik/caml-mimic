import argparse
import json
import sys
from typing import Any, Dict


import numpy as np
from constants import MAX_LENGTH
from learn import tools

from gevent.pywsgi import WSGIServer


import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from flask import Flask, request
import flask
from flask_cors import CORS, cross_origin

import bentoml

# HADMID 120233
TEXT = "admission date discharge date date of birth sex m service surgery allergies patient recorded as having no known allergies to drugs attending first name3 lf chief complaint hand fracture major surgical or invasive procedure none history of present illness y o man s p motorcycle crash vs car the patient was reportedly wearing a helmet however it was found feet from the pt at the scene the pt was found unresponsive by paramedics at the scene with uneven pupils after bag mask ventilation the pt became responsive with equal and reactive pupils he then became highly combative his gcs was however he was intubated for severe agitation and transported to osh he was then transported to the ed at hospital1 and found to have 2mm punctate hemorrhages in his left frontal lobe per the ct report from the osh pt also sustained a left wrist fracture and lacerations to his hands the patient remains intubated and sedated past medical history unknown social history n c family history n c physical exam ra nad aaox3 rrr ctab abd soft nt nd bs pertinent results wbc rbc hgb hct mcv mch mchc rdw plt ct pt ptt inr pt fibrino urean creat lipase asa neg ethanol neg acetmnp neg bnzodzp neg barbitr neg tricycl neg type art temp po2 pco2 ph caltco2 base xs intubat intubated lactate wbc rbc hgb hct mcv mch mchc rdw plt ct glucose urean creat na k cl hco3 angap calcium phos mg brief hospital course 21m mcc vs car and intubated in the field due to aggitation was admitted on hospital1 on discharge medications acetaminophen mg tablet sig two tablet po q4h every hours as needed discharge disposition home discharge diagnosis small punctate intracranial hemorrhages in the frontal lobe 2nd and 3rd metacarpal fracture of the left hand degloving injury of right hand discharge condition patient is hemodynamically stable tolerated a regular diet with normal and stable vital signs discharge instructions go to the emergency department or see your own doctor right away if any problems develop including the following your pain gets worse you develop pain numbness tingling or weakness in your arms or legs you lose control of your bowels or urine passing water trouble walking your pain is not getting better after days anything else that worries you regarding your wounds please present to the ed if watch carefully for signs of infection redness warmth increasing pain swelling drainage of pus thick white yellow or green liquid or fevers if you have numbness pins and needles or pain in the area of your injury the stitches are loose or the wound is opening up followup instructions regarding your left hand fractures and neck follow up with orthospine dr last name stitle in orthopedics in two weeks please call telephone fax to make an appointment regarding your hand injuries please follow up with plastic surgery this tuesday in their hand clinic telephone fax completed by"


def main(args):

    with open(args.dicts, "r") as f:
        dicts = json.load(f)
        dicts["ind2w"] = {int(k): v for k, v in dicts["ind2w"].items()}  # stupid json
        dicts["ind2c"] = {int(k): v for k, v in dicts["ind2c"].items()}  # stupid json

    with open(args.label_description, "r") as f:
        label_desc = json.load(f)

    model = tools.pick_model(args, dicts)

    model.eval()
    w2ind, ind2c = dicts["w2ind"], dicts["ind2c"]
    if args.save_model:
        dicts["label_desc"] = label_desc
        saved_model = bentoml.pytorch.save_model("caml", model, custom_objects=dicts)
        print(saved_model)
        


    def get_preds(txt, k):
        text = [
            int(w2ind[w] + 1) if w in w2ind else len(w2ind) + 1 for w in txt.split()
        ]  # OOV words are given a unique index at end of vocab lookup
        # truncate long documents
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH]
        else:
            text.extend([0] * (MAX_LENGTH - len(text)))

        data = np.array(text)[np.newaxis, ...]
        print(data)
        # assert False
        data, target = Variable(torch.tensor(data, dtype=torch.int32), volatile=True), None
        if args.gpu:
            data = data.cuda()
        model.zero_grad()
        output, *_ = model(data, target)

        output = F.sigmoid(output)
        output = output.data.cpu().numpy().squeeze()

        # top-k predictions:
        top_k_indices = np.argsort(output)[::-1][:k]

        return {
            "text": txt,
            "result": [
                {
                    "code": ind2c[i],
                    "name": label_desc.get(ind2c[i], "Unknown description..."),
                    "probability": float(output[i]),
                }
                for i in top_k_indices
            ],
        }

    if args.test:
        print(get_preds(TEXT, 5))
        return

    app = Flask(__name__)
    CORS(app)

    @app.route("/", methods=["POST", "OPTIONS"])
    @cross_origin()
    def predict():

        data: Dict[str, Any] = request.get_json()
        print(data)
        if not data:
            return {"error": "Something, somewhere went horribly wrong."}
        txt = data.get("text", TEXT)

        try:
            k = int(data.get("k", 5))
        except:
            k = 5

        res = get_preds(txt, k)
        response = flask.jsonify(res)
        # response.headers.add("Access-Control-Allow-Origin", "*")
        # response.headers.add(
        #     "Access-Control-Allow-Headers",
        #     "Origin, X-Requested-With, Content-Type, Accept",
        # )
        return response

    print("loading server")
    server = WSGIServer(("", 5078), app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")

    parser.add_argument(
        "model",
        type=str,
        choices=[
            "cnn_vanilla",
            "rnn",
            "conv_attn",
            "multi_conv_attn",
            "logreg",
            "saved",
        ],
        help="model",
    )
    parser.add_argument("dicts", type=str, help="Word/label index dictionaries.")
    parser.add_argument("label_description", type=str, help="Label to description dict.")
    parser.add_argument(
        "--test",
        dest="test",
        action="store_const",
        required=False,
        const=True,
        help="just a debug thing",
    )
    parser.add_argument(
        "--embed-file",
        type=str,
        required=False,
        dest="embed_file",
        help="path to a file holding pre-trained embeddings",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        choices=["lstm", "gru"],
        help="what kind of RNN to use (default: GRU)",
        dest="cell_type",
        default="gru",
    )
    parser.add_argument(
        "--rnn-dim",
        type=int,
        required=False,
        dest="rnn_dim",
        default=128,
        help="size of rnn hidden layer (default: 128)",
    )
    parser.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_const",
        required=False,
        const=True,
        help="optional flag for rnn to use a bidirectional model",
    )
    parser.add_argument(
        "--rnn-layers",
        type=int,
        required=False,
        dest="rnn_layers",
        default=1,
        help="number of layers for RNN models (default: 1)",
    )
    parser.add_argument(
        "--embed-size",
        type=int,
        required=False,
        dest="embed_size",
        default=100,
        help="size of embedding dimension. (default: 100)",
    )
    parser.add_argument(
        "--filter-size",
        type=str,
        required=False,
        dest="filter_size",
        default=4,
        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5",
    )
    parser.add_argument(
        "--num-filter-maps",
        type=int,
        required=False,
        dest="num_filter_maps",
        default=50,
        help="size of conv output (default: 50)",
    )
    parser.add_argument(
        "--pool",
        choices=["max", "avg"],
        required=False,
        dest="pool",
        help="which type of pooling to do (logreg model only)",
    )
    parser.add_argument(
        "--code-emb",
        type=str,
        required=False,
        dest="code_emb",
        help="point to code embeddings to use for parameter initialization, if applicable",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=False,
        dest="weight_decay",
        default=0,
        help="coefficient for penalizing l2 norm of model weights (default: 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        dest="lr",
        default=1e-3,
        help="learning rate for Adam optimizer (default=1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        dest="batch_size",
        default=16,
        help="size of training batches",
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        required=False,
        default=0.5,
        help="optional specification of dropout (default: 0.5)",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        required=False,
        dest="lmbda",
        default=0,
        help="hyperparameter to tradeoff BCE loss and similarity embedding loss. defaults to 0, which won't create/use the description embedding module at all. ",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mimic2", "mimic3"],
        dest="version",
        default="mimic3",
        required=False,
        help="version of MIMIC in use (default: mimic3)",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        dest="test_model",
        required=False,
        help="path to a saved model to load and evaluate",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="f1_micro",
        required=False,
        dest="criterion",
        help="which metric to use for early stopping (default: f1_micro)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        required=False,
        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        required=False,
        const=True,
        help="optional flag to use GPU if available",
    )
    parser.add_argument(
        "--public-model",
        dest="public_model",
        action="store_const",
        required=False,
        const=True,
        help="optional flag for testing pre-trained models from the public github",
    )
    parser.add_argument(
        "--stack-filters",
        dest="stack_filters",
        action="store_const",
        required=False,
        const=True,
        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them",
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        action="store_const",
        required=False,
        const=True,
        help="optional flag to save samples of good / bad predictions",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_const",
        required=False,
        const=True,
        help="optional flag not to print so much during training",
    )
    parser.add_argument(
        "--save-model",
        dest="save_model",
        action="store_const",
        required=False,
        const=True,
        help="optional flag not to print so much during training",
    )

    args = parser.parse_args()
    command = " ".join(["python"] + sys.argv)
    args.command = command
    main(args)
