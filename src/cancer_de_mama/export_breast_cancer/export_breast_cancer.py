from pathlib import Path

import joblib


class ExportBreastCancer:
    """Exporta o modelo treinado para ser carregado em previsões futuras."""

    def __init__(self, training_and_evaluation, output_dir=None):
        self.training_and_evaluation = training_and_evaluation
        self.output_dir = Path(
            output_dir
            if output_dir is not None
            else training_and_evaluation.output_dir
        )

    def execute(self):
        print("\n--------- Exportação do modelo -----------------------------------------------------\n")
        evaluator = self.training_and_evaluation
        if evaluator.final_model is None:
            raise ValueError("Treine e selecione o modelo antes de exportá-lo.")
        if evaluator.selected_model_name is None or evaluator.selected_threshold is None:
            raise ValueError("O nome do modelo e o limiar de classificação são necessários.")
        if evaluator.modeling is None or evaluator.modeling.X_train is None:
            raise ValueError("As características usadas no treinamento não estão disponíveis.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": evaluator.final_model,
            "model_name": evaluator.selected_model_name,
            "threshold": float(evaluator.selected_threshold),
            "feature_names": list(evaluator.modeling.X_train.columns),
        }
        model_path = self.output_dir / "breast_cancer_model.joblib"
        joblib.dump(artifact, model_path)
        print(f"Modelo exportado para previsões futuras: {model_path}")
        return model_path
