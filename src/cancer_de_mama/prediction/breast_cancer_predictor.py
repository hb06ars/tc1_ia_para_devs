from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BreastCancerPredictor:
    """Carrega o modelo salvo e classifica novos registros informados pelo usuário."""

    def __init__(self, model_path="results/breast_cancer_model.joblib"):
        artifact = joblib.load(Path(model_path))
        required_keys = {"model", "model_name", "threshold", "feature_names"}
        if not isinstance(artifact, dict):
            raise ValueError("O arquivo não contém um artefato de modelo válido.")
        missing_keys = required_keys.difference(artifact)
        if missing_keys:
            raise ValueError(f"O arquivo do modelo não contém os campos: {sorted(missing_keys)}")

        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.threshold = float(artifact["threshold"])
        self.feature_names = list(artifact["feature_names"])
        if not self.feature_names:
            raise ValueError("O arquivo do modelo não contém nomes de características.")

    def predict_one(self, features):
        """Valida as 30 características e retorna diagnóstico e score do modelo."""
        if not isinstance(features, dict):
            raise TypeError("Informe as características em um dicionário.")

        missing_features = set(self.feature_names).difference(features)
        extra_features = set(features).difference(self.feature_names)
        if missing_features or extra_features:
            problems = []
            if missing_features:
                problems.append(f"faltando: {sorted(missing_features)}")
            if extra_features:
                problems.append(f"não reconhecidas: {sorted(extra_features)}")
            raise ValueError("Características inválidas (" + "; ".join(problems) + ").")

        values = {}
        for name in self.feature_names:
            try:
                value = float(str(features[name]).strip().replace(",", "."))
            except (TypeError, ValueError) as error:
                raise ValueError(f"O valor de '{name}' precisa ser numérico.") from error
            if not np.isfinite(value):
                raise ValueError(f"O valor de '{name}' precisa ser finito.")
            values[name] = value

        input_frame = pd.DataFrame([values], columns=self.feature_names)
        if hasattr(self.model, "predict_proba"):
            classifier = self.model[-1]
            positive_class_index = list(classifier.classes_).index(1)
            score = float(self.model.predict_proba(input_frame)[0, positive_class_index])
            score_type = "probability"
        else:
            score = float(self.model.decision_function(input_frame)[0])
            score_type = "decision_score"

        diagnosis = "Maligno" if score >= self.threshold else "Benigno"
        return {
            "diagnosis": diagnosis,
            "model": self.model_name,
            "score": score,
            "score_type": score_type,
            "threshold": self.threshold,
        }
