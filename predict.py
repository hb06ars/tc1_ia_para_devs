import math
from pathlib import Path

from src.cancer_de_mama.prediction.breast_cancer_predictor import BreastCancerPredictor


def main():
    model_path = Path("results/breast_cancer_model.joblib")
    predictor = BreastCancerPredictor(model_path)

    print("\nPrevisão experimental de câncer de mama")
    print(f"Modelo carregado: {predictor.model_name}")
    print("Digite os valores das características do registro.")
    print("Use ponto ou vírgula como separador decimal.\n")

    features = {}
    for feature in predictor.feature_names:
        while True:
            value = input(f"{feature}: ").strip()
            try:
                numeric_value = float(value.replace(",", "."))
                if not math.isfinite(numeric_value):
                    raise ValueError
                features[feature] = numeric_value
                break
            except ValueError:
                print("Valor inválido. Digite um número finito.")

    result = predictor.predict_one(features)
    print(f"\nClassificação do modelo: {result['diagnosis']}")
    if result["score_type"] == "probability":
        print(f"Score do modelo: {result['score']:.1%}")
    else:
        print(f"Score de decisão do SVM: {result['score']:.4f}")
    print("\nEsta demonstração acadêmica não é um diagnóstico médico.")


if __name__ == "__main__":
    main()
