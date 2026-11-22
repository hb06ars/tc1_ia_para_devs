from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


class ModelTrainingAndEvaluation:
    """
    Treina os modelos, seleciona o melhor usando validação e faz uma avaliação
    final no conjunto de teste, incluindo métricas e explicabilidade.
    """

    POSITIVE_CLASS = 1
    RECALL_BETA = 2
    RANDOM_STATE = 42

    def __init__(self, modeling=None, output_dir="results"):
        self.modeling = modeling
        self.output_dir = Path(output_dir)
        self.graphics_dir = self.output_dir / "graphics"
        self.metrics_dir = self.output_dir / "metrics"
        self.selected_model_name = None
        self.final_model = None
        self.selected_threshold = None
        self.cross_validation_results = None
        self.test_metrics = None

    def execute(self, modeling=None):
        print("\n--------- Treinamento e Avaliação do modelo ----------------------------------------\n")

        """ Validação de modelo not null e criação de diretório se necessário."""
        self.validation_exists_modeling_and_dir(modeling)

        """ Compara os modelos com validação cruzada estratificada de 5 partes. """
        self.cross_validation_results = self.cross_validate_models()

        """ Escolhe o modelo pela validação cruzada e calibra seu limiar. """
        self.select_model_and_refit()

        """ Avalia o modelo escolhido no conjunto de teste e calcula as métricas finais 
            Gerando: (Feature importance, Matriz de Confusão, etc). """
        self.test_metrics = self.evaluate_on_test()

        """ Feature Importance: Mostra quais características ajudam mais a identificar casos malignos. """
        self.analyze_feature_importance()

        """ SHAP: Cria gráficos para mostrar quais características influenciaram as previsões."""
        self.run_shap_analysis()

        """ Discussão dos resultados finais."""
        self.discuss_results()

        return self

    def validation_exists_modeling_and_dir(self, modeling):
        """ Validando modeling"""
        self.modeling = modeling if modeling is not None else self.modeling
        if self.modeling is None or not self.modeling.models:
            raise ValueError("Execute a etapa de modelagem antes da avaliação.")

        """ Validando diretório"""
        self.graphics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def cross_validate_models(self):
        """Compara os modelos em cinco rodadas usando apenas os dados de desenvolvimento."""
        results = []
        print("=== VALIDAÇÃO CRUZADA ESTRATIFICADA (5 PARTES) ===")

        X_development = pd.concat(
            [self.modeling.X_train, self.modeling.X_validation],
            ignore_index=True,
        )
        y_development = pd.concat(
            [self.modeling.y_train, self.modeling.y_validation],
            ignore_index=True,
        )
        cross_validator = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.RANDOM_STATE,
        )

        for name, model in self.modeling.models.items():
            fold_metrics = []
            out_of_fold_scores = np.empty(len(X_development))

            for train_indices, validation_indices in cross_validator.split(
                X_development,
                y_development,
            ):
                fold_model = clone(model)
                X_fold_train = X_development.iloc[train_indices]
                y_fold_train = y_development.iloc[train_indices]
                X_fold_validation = X_development.iloc[validation_indices]
                y_fold_validation = y_development.iloc[validation_indices]

                fold_model.fit(X_fold_train, y_fold_train)
                fold_predictions = fold_model.predict(X_fold_validation)
                fold_metrics.append(
                    self.calculate_metrics(y_fold_validation, fold_predictions)
                )
                out_of_fold_scores[validation_indices] = self.get_positive_class_scores(
                    fold_model,
                    X_fold_validation,
                )

            fold_metrics_frame = pd.DataFrame(fold_metrics)
            metrics = {
                f"{metric}_mean": fold_metrics_frame[metric].mean()
                for metric in ("accuracy", "precision", "recall", "f1_score")
            }
            metrics.update(
                {
                    f"{metric}_std": fold_metrics_frame[metric].std(ddof=0)
                    for metric in ("accuracy", "precision", "recall", "f1_score")
                }
            )

            threshold, threshold_predictions = self.find_recall_focused_threshold(
                out_of_fold_scores,
                y_development,
            )
            threshold_metrics = self.calculate_metrics(
                y_development,
                threshold_predictions,
            )
            metrics.update(
                {
                    f"oof_{metric}": value
                    for metric, value in threshold_metrics.items()
                }
            )
            metrics["model"] = name
            metrics["threshold"] = threshold
            results.append(metrics)

        # A escolha prioriza a média do recall nas cinco partes; os demais critérios
        # ajudam no desempate e a precision mostra a frequência de falsos positivos.
        results_frame = pd.DataFrame(results).sort_values(
            by=["recall_mean", "precision_mean", "f1_score_mean", "accuracy_mean"],
            ascending=False,
        )

        results_frame.to_csv(
            self.metrics_dir / "comparacao_modelos_validacao_cruzada.csv",
            index=False,
        )

        print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print(
            "\nAs métricas *_mean e *_std resumem os cinco folds. "
            "O limiar é calibrado pelas previsões fora do fold, priorizando recall "
            "com F2-score (recall tem peso maior que precision)."
        )
        return results_frame

    def select_model_and_refit(self):
        """Escolhe o modelo pela validação cruzada e treina com todos os dados de desenvolvimento.
        'Prioriza recall maligno': quer dizer que, ao escolher o limiar,
        damos mais importância a identificar o maior número possível de casos que realmente são malignos.

        Isso reduz a chance de o modelo classificar um caso maligno como benigno — um falso negativo —
        mas pode aumentar os falsos positivos, isto é, casos benignos sinalizados como malignos.
        Por isso, “priorizar” não significa garantir que falsos negativos não acontecerão.
        """

        best_result = self.cross_validation_results.iloc[0]
        self.selected_model_name = best_result["model"]
        self.final_model = clone(self.modeling.models[self.selected_model_name])
        self.selected_threshold = best_result["threshold"]
        X_development = pd.concat(
            [self.modeling.X_train, self.modeling.X_validation],
            ignore_index=True,
        )
        y_development = pd.concat(
            [self.modeling.y_train, self.modeling.y_validation],
            ignore_index=True,
        )
        self.final_model.fit(X_development, y_development)
        print(f"\nModelo selecionado pela validação: {self.selected_model_name}")
        print(
            f"Limiar escolhido pelo F{self.RECALL_BETA}-score "
            f"(prioriza recall maligno): {self.selected_threshold:.4f}"
        )
        print(f"Modelo final treinado com {len(X_development)} registros de desenvolvimento.")

    def evaluate_on_test(self):
        """Avalia o modelo escolhido no conjunto de teste e calcula as métricas finais (Feature importance, Matriz de Confusão, etc)."""

        y_true = self.modeling.y_test
        predictions = self.predict_with_threshold(
            self.final_model,
            self.modeling.X_test,
            self.selected_threshold,
        )
        metrics = self.calculate_metrics(y_true, predictions)
        metrics["model"] = self.selected_model_name
        metrics["threshold"] = self.selected_threshold

        print(f"\n======================================= AVALIAÇÃO FINAL NO TESTE =======================================")
        print(f"Teste final com : {self.selected_model_name}")

        for metric in ("accuracy", "precision", "recall", "f1_score"):
            print(f"{metric}: {metrics[metric]:.3f}")
        print("\nRelatório por diagnóstico (1 = maligno):")


        """ Feature importance: variáveis mais importantes (recall maligno)."""
        print(
            classification_report(
                y_true,
                predictions,
                labels=[0, 1],
                target_names=["Benigno", "Maligno"],
                zero_division=0,
            )
        )

        pd.DataFrame([metrics]).to_csv(
            self.metrics_dir / "metricas_teste.csv",
            index=False,
        )

        """ Matriz de Confusão"""
        self.plot_confusion_matrix(y_true, predictions)
        return metrics

    def calculate_metrics(self, y_true, predictions):
        return {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, pos_label=self.POSITIVE_CLASS, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, pos_label=self.POSITIVE_CLASS, zero_division=0
            ),
            "f1_score": f1_score(
                y_true, predictions, pos_label=self.POSITIVE_CLASS, zero_division=0
            ),
        }

    def get_positive_class_scores(self, model, features):
        """Obtém um score para a classe maligna, seja probabilidade ou margem do SVM."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]
        return model.decision_function(features)

    def predict_with_threshold(self, model, features, threshold):
        scores = self.get_positive_class_scores(model, features)
        return (scores >= threshold).astype(int)

    def find_recall_focused_threshold(self, scores, labels):
        """Escolhe o limiar pelo F2-score, que dá mais peso ao recall."""
        best_threshold = None
        best_predictions = None
        best_metrics = None

        for threshold in np.unique(scores):
            predictions = (scores >= threshold).astype(int)
            metrics = self.calculate_metrics(labels, predictions)
            rank = (
                fbeta_score(
                    labels,
                    predictions,
                    beta=self.RECALL_BETA,
                    pos_label=self.POSITIVE_CLASS,
                    zero_division=0,
                ),
                metrics["recall"],
                metrics["precision"],
                metrics["f1_score"],
                threshold,
            )
            if best_metrics is None or rank > best_metrics:
                best_threshold = threshold
                best_predictions = predictions
                best_metrics = rank

        return best_threshold, best_predictions

    def plot_confusion_matrix(self, y_true, predictions):
        """ Matriz de Confusão"""

        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benigno", "Maligno"],
            yticklabels=["Benigno", "Maligno"],
        )
        plt.title(f"Matriz de Confusão - {self.selected_model_name}")
        plt.xlabel("Diagnóstico previsto")
        plt.ylabel("Diagnóstico real")
        plt.tight_layout()
        plt.savefig(self.graphics_dir / "matriz_confusao_teste.png", dpi=300)
        plt.close()

    def analyze_feature_importance(self):
        """ Feature Importance: Mostra quais características ajudam mais a identificar casos malignos. """

        def recall_at_selected_threshold(model, features, labels):
            predictions = self.predict_with_threshold(
                model,
                features,
                self.selected_threshold,
            )
            return recall_score(labels, predictions, pos_label=self.POSITIVE_CLASS, zero_division=0)

        importance = permutation_importance(
            self.final_model,
            self.modeling.X_test,
            self.modeling.y_test,
            scoring=recall_at_selected_threshold,
            n_repeats=10,
            random_state=self.RANDOM_STATE,
        )
        importance_frame = pd.DataFrame(
            {
                "feature": self.modeling.X_test.columns,
                "importance": importance.importances_mean,
                "std": importance.importances_std,
            }
        ).sort_values("importance", ascending=False)
        importance_frame.to_csv(
            self.metrics_dir / "importancia_caracteristicas.csv",
            index=False,
        )
        top_features = importance_frame.head(15).sort_values("importance")

        plt.figure(figsize=(10, 7))
        sns.barplot(data=top_features, x="importance", y="feature", color="steelblue")
        plt.title("Importância das variáveis por permutação (recall maligno)")
        plt.xlabel("Queda média no recall ao embaralhar a variável")
        plt.ylabel("Variável")
        plt.tight_layout()
        plt.savefig(self.graphics_dir / "importancia_caracteristicas.png", dpi=300)
        plt.close()

        print("\nFeature Importance: 10 Variáveis mais importantes por permutação (recall maligno):")
        print(importance_frame.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    def run_shap_analysis(self):
        """Cria gráficos (SHAP) para mostrar quais características influenciaram as previsões."""
        feature_names = self.modeling.X_test.columns
        background = self.modeling.X_train.sample(
            n=min(30, len(self.modeling.X_train)),
            random_state=self.RANDOM_STATE,
        )
        examples = self.modeling.X_test.head(30)

        def model_score(values):
            values_frame = pd.DataFrame(values, columns=feature_names)
            if hasattr(self.final_model, "predict_proba"):
                return self.final_model.predict_proba(values_frame)[:, 1]
            return self.final_model.decision_function(values_frame)

        explainer = shap.Explainer(
            model_score,
            background,
            algorithm="permutation",
        )
        shap_values = explainer(
            examples,
            max_evals=2 * len(feature_names) + 1,
        )

        plt.figure(figsize=(11, 7))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("SHAP - contribuição das variáveis para as previsões")
        plt.tight_layout()
        plt.savefig(self.graphics_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 7))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("SHAP - importância global das variáveis")
        plt.tight_layout()
        plt.savefig(self.graphics_dir / "shap_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nGráficos SHAP salvos em {self.graphics_dir}.")

    def discuss_results(self):
        print(
            "\nInterpretação: recall mede quantos diagnósticos malignos reais foram "
            "identificados. É uma métrica importante neste problema, pois falsos "
            "negativos podem atrasar uma investigação médica. O F1-score equilibra "
            "recall e precisão, enquanto accuracy mostra a proporção geral de acertos."
        )
        print(
            "Limitação: a validação cruzada e o teste usam apenas este dataset e não "
            "demonstram segurança clínica em pacientes reais. O modelo é apenas apoio "
            "à análise; não substitui exames, avaliação profissional ou a decisão do médico."
        )
