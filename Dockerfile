FROM continuumio/miniconda3

WORKDIR /usr/share/caml

COPY . /usr/share/caml

ADD ./predictions/reproCAML_mimic3_full/model.pth /usr/share/caml/models/reproCAML_mimic3_full/model.pth

RUN conda env update --file environment.yml

# SHELL ["conda", "run", "-n", "caml", "/bin/bash"]

#RUN conda activate caml

RUN conda run -n caml pip install https://download.pytorch.org/whl/cu90/torch-0.3.0-cp36-cp36m-linux_x86_64.whl

ENV PYTHONPATH=/usr/share/caml

EXPOSE 5078

RUN conda run -n caml pip install flask_cors bentoml

CMD ["conda", "run", "-n", "caml", "--no-capture-output", "python", "learn/cli.py", "conv_attn", "dicts.json", "label_desc.json", "--filter-size", "10", "--num-filter-maps", "50", "--test-model", "./models/reproCAML_mimic3_full/model.pth"]