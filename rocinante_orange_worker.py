import json
import os
import pickle
import sys
import traceback

import numpy as np
import Orange
from orangecontrib.imageanalytics.squeezenet_model import SqueezenetModel
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "rocinanteAI.pkcls")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    image_loader = ImageLoader()
    embedder = SqueezenetModel()
    print(json.dumps({"ready": True}), flush=True)
except Exception as exc:
    print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
    traceback.print_exc(file=sys.stderr)
    raise SystemExit(1)

for line in sys.stdin:
    try:
        request = json.loads(line)
        image = image_loader.load_image_or_none(request["image"], (227, 227))
        if image is None:
            raise ValueError("Could not read the generated spectrogram image.")
        embedding = embedder.predict(embedder.preprocess(image)).reshape(-1)
        audio = np.asarray(request["audio"], dtype=float).reshape(1, -1)
        if audio.shape[1] != 23:
            raise ValueError(f"Expected 23 audio features, received {audio.shape[1]}.")
        x = np.concatenate((audio, embedding.astype(float, copy=False).reshape(1, -1)), axis=1)
        if x.shape[1] != len(model.original_domain.attributes):
            raise ValueError(f"Expected {len(model.original_domain.attributes)} model inputs, received {x.shape[1]}.")
        metas = np.full((1, len(model.original_domain.metas)), None, dtype=object)
        data = Orange.data.Table.from_numpy(
            model.original_domain, x, Y=np.array([np.nan]), metas=metas
        )
        prediction = model(data)
        label = model.domain.class_var.str_val(float(np.asarray(prediction).reshape(-1)[0]))
        print(json.dumps({"label": label}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        traceback.print_exc(file=sys.stderr)
