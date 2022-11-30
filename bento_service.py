import os
from typing import Optional
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel


import numpy as np


import torch.nn.functional as F

import torch
from torch.autograd import Variable

import bentoml

MAX_LENGTH = 2500


TEXT = "admission date discharge date date of birth sex m service surgery allergies patient recorded as having no known allergies to drugs attending first name3 lf chief complaint hand fracture major surgical or invasive procedure none history of present illness y o man s p motorcycle crash vs car the patient was reportedly wearing a helmet however it was found feet from the pt at the scene the pt was found unresponsive by paramedics at the scene with uneven pupils after bag mask ventilation the pt became responsive with equal and reactive pupils he then became highly combative his gcs was however he was intubated for severe agitation and transported to osh he was then transported to the ed at hospital1 and found to have 2mm punctate hemorrhages in his left frontal lobe per the ct report from the osh pt also sustained a left wrist fracture and lacerations to his hands the patient remains intubated and sedated past medical history unknown social history n c family history n c physical exam ra nad aaox3 rrr ctab abd soft nt nd bs pertinent results wbc rbc hgb hct mcv mch mchc rdw plt ct pt ptt inr pt fibrino urean creat lipase asa neg ethanol neg acetmnp neg bnzodzp neg barbitr neg tricycl neg type art temp po2 pco2 ph caltco2 base xs intubat intubated lactate wbc rbc hgb hct mcv mch mchc rdw plt ct glucose urean creat na k cl hco3 angap calcium phos mg brief hospital course 21m mcc vs car and intubated in the field due to aggitation was admitted on hospital1 on discharge medications acetaminophen mg tablet sig two tablet po q4h every hours as needed discharge disposition home discharge diagnosis small punctate intracranial hemorrhages in the frontal lobe 2nd and 3rd metacarpal fracture of the left hand degloving injury of right hand discharge condition patient is hemodynamically stable tolerated a regular diet with normal and stable vital signs discharge instructions go to the emergency department or see your own doctor right away if any problems develop including the following your pain gets worse you develop pain numbness tingling or weakness in your arms or legs you lose control of your bowels or urine passing water trouble walking your pain is not getting better after days anything else that worries you regarding your wounds please present to the ed if watch carefully for signs of infection redness warmth increasing pain swelling drainage of pus thick white yellow or green liquid or fevers if you have numbness pins and needles or pain in the area of your injury the stitches are loose or the wound is opening up followup instructions regarding your left hand fractures and neck follow up with orthospine dr last name stitle in orthopedics in two weeks please call telephone fax to make an appointment regarding your hand injuries please follow up with plastic surgery this tuesday in their hand clinic telephone fax completed by"


TAG = os.environ.get("CAML_TAG_ARG", "latest")
print(TAG)
assert TAG != "caml:hv2545dlt6zj5h5i"

model = bentoml.pytorch.get(TAG)
print(model)
dicts = model.custom_objects
w2ind, ind2c, label_desc = dicts["w2ind"], dicts["ind2c"], dicts["desc"]

caml_runner: bentoml.Runner = model.to_runner()
print(caml_runner.models[0])
svc = bentoml.Service("caml_classifier", runners=[caml_runner])



class InputFeatures(BaseModel):
    text: str
    k: Optional[int] = 5



@svc.api(input=JSON(pydantic_model=InputFeatures), output=JSON())
def predict(inputs: InputFeatures):
    text = [
        int(w2ind[w] + 1) if w in w2ind else len(w2ind) + 1 for w in inputs.text.split()
    ]  # OOV words are given a unique index at end of vocab lookup
    # truncate long documents
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    else:
        text.extend([0] * (MAX_LENGTH - len(text)))

    data = np.array(text)[np.newaxis, ...]
    print(data)

    data = Variable(torch.LongTensor(data), volatile=True)
    
    with torch.no_grad():
        output, *_ = caml_runner.run(data)

    output = F.sigmoid(output)
    output = output.data.cpu().numpy().squeeze()

    # top-k predictions:
    top_k_indices = np.argsort(output)[::-1][: inputs.k]

    return {
        "text": inputs.text,
        "result": [
            {
                "code": ind2c[i],
                "name": label_desc.get(ind2c[i], "Unknown description..."),
                "probability": float(output[i]),
            }
            for i in top_k_indices
        ],
    }
    


if __name__ == "__main__":
    ...
