# biomodule.py
import pickle
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "data" / "her2_rf_model.pkl"


class PhytoHER2Module:
    def __init__(self, model_path: str | Path | None = None):
        # Load your trained model from the data folder
        target_path = model_path or (MODEL_PATH if MODEL_PATH.exists() else "data/her2_rf_model.pkl")
        with open(target_path, "rb") as f:
            self.model = pickle.load(f)

    def execute(self, inputs, *, context=None):
        """
        Biosimulant runtime execution hook.
        Expects 'inputs' to contain molecular fingerprint data.
        Returns BioSignal payload with HER2 inhibition probability.
        """
        features = np.array(inputs["fingerprint"]).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]

        return {
            "prediction_class": int(prediction),
            "inhibition_probability": float(probability),
            "status": "COMPLETED",
        }
