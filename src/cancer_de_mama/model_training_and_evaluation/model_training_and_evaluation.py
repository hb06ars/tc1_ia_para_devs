from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelTrainingAndEvaluation:
    """
    Treina os modelos, seleciona o melhor usando validação e faz uma avaliação
    final no conjunto de teste, incluindo métricas e explicabilidade.
    """

    POSITIVE_CLASS = 1
    RANDOM_STATE = 42

    def __init__(self, modeling=None, output_dir="results"):
        self.modeling = modeling
        self.output_dir = Path(output_dir)
        self.graphics_dir = self.output_dir / "graphics"
        self.metrics_dir = self.output_dir / "metrics"
        self.selected_model_name = None
        self.final_model = None
        self.validation_results = None
        self.test_metrics = None

    def execute(self, modeling=None):
        print("\n--------- Treinamento e Avaliação do modelo ----------------------------------------\n")

        """ Validação de modelo not null e criação de diretório se necessário."""
        self.validation_exists_modeling_and_dir(modeling)

        """ Treina cada candidato, no teste de validação (candidato = tipo de modelo: Regressão Logística, SVM, Árvore de Decisão) 
            e compara os seus resultados. """
        self.validation_results = self.train_and_compare_on_validation()

        """ Pega o 'best' (melhor no teste de validação foi usada para escolher o melhor modelo).
            No nosso caso, foi Regressao Logística e 're-treina'. 
            Assim, o modelo final aprende com mais exemplos.
        """
        self.select_and_refit_model()

        """ Calcula as métricas finais no teste, que não participa da seleção do modelo.
            Assim, o teste dá uma estimativa mais imparcial de como o modelo pode se sair 
            dos dados usados para desenvolvê-lo.
        """
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

    def train_and_compare_on_validation(self):
        results = []
        print("=== Treino e comparação na validação ===")

        """ 
            Cada modelo aprende padrões usando somente os dados de treino.
            Em seguida, predictions guarda as classes previstas (resultados corretos previstos) para os dados de validação.
            
            calculate_metrics compara essas previsões com os diagnósticos reais da validação:
                - accuracy: proporção total de previsões corretas, entre benignos e malignos.
                - precision: entre os casos previstos como malignos, quantos realmente são malignos.
                - recall: entre os casos realmente malignos, quantos foram identificados pelo modelo.
                
                - f1_score: equilíbrio entre precision e recall, útil quando os dois importam.
                    Exemplo: suponha que o modelo identifique 10 casos como malignos, e 8 realmente sejam malignos (precision = 80%). 
                    Mas havia 20 casos malignos no total, então ele encontrou apenas 8 (recall = 40%). 
                    O F1-score será cerca de 53%, refletindo que o modelo foi bem em precision, mas deixou passar vários casos malignos.
            """
        for name, model in self.modeling.models.items():
            model.fit(self.modeling.X_train, self.modeling.y_train)
            predictions = model.predict(self.modeling.X_validation)
            metrics = self.calculate_metrics(self.modeling.y_validation, predictions)
            metrics["model"] = name
            results.append(metrics)

        """
            Ordena os modelos do maior para o menor recall; em caso de empate,
            usa F1-score e accuracy como critérios seguintes. 
            Depois, salva e exibe a comparação.
            Priorizamos o recall maligno porque casos malignos não identificados (falsos negativos)
            podem atrasar a investigação. Falsos positivos também têm consequências e podem
            levar a exames adicionais; a previsão do modelo não substitui a avaliação médica.
        """
        results_frame = pd.DataFrame(results).sort_values(
            by=["recall", "f1_score", "accuracy"],
            ascending=False,
        )

        results_frame.to_csv(self.metrics_dir / "comparacao_modelos_validacao.csv", index=False)

        print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print(
            "\nA seleção prioriza o recall da classe maligna: quanto maior, "
            "menor a proporção de casos malignos que o modelo deixa passar."
        )
        return results_frame

    def select_and_refit_model(self):
        """ Pega o melhor treino e 'retreina' """
        best_result = self.validation_results.iloc[0]
        self.selected_model_name = best_result["model"]
        self.final_model = clone(self.modeling.models[self.selected_model_name])

        X_train_validation = pd.concat(
            [self.modeling.X_train, self.modeling.X_validation],
            ignore_index=True,
        )
        y_train_validation = pd.concat(
            [self.modeling.y_train, self.modeling.y_validation],
            ignore_index=True,
        )
        self.final_model.fit(X_train_validation, y_train_validation)
        print(f"\nModelo selecionado pela validação: {self.selected_model_name}")
        print(f"Foi treinado novamente com {len(X_train_validation)} registros de treino + validação.")

    def evaluate_on_test(self):
        """
            Calcula as métricas finais no teste, que não participa da seleção do modelo.
            Assim, o teste dá uma estimativa mais imparcial de como o modelo pode se sair fora
            dos dados usados para desenvolvê-lo.
        """

        y_true = self.modeling.y_test
        predictions = self.final_model.predict(self.modeling.X_test)
        metrics = self.calculate_metrics(y_true, predictions)
        metrics["model"] = self.selected_model_name

        print(f"\n======================================= AVALIAÇÃO FINAL NO TESTE =======================================")
        print(f"Teste final com : {self.selected_model_name}")

        """ 
        • Benigno: 
                havia 54 casos. 
                recall = 1,00 indica que o modelo identificou todos como benignos.
                precision = 0,98 indica que, entre os casos previstos como benignos, cerca de 98% eram realmente benignos.
                
        • Maligno: 
                havia 32 casos. 
                recall = 0,97 indica que o modelo identificou 31 dos 32 casos malignos e deixou 1 passar. 
                precision = 1,00 indica que todos os casos que previu como malignos eram realmente malignos.
                
        • F1-score: 
            combina precision e recall em uma medida. É alto quando as duas métricas também são altas.
        
        • Accuracy = 0,99: o modelo acertou 85 dos 86 casos, aproximadamente 98,8%.
        
        • Macro avg: média simples das métricas das duas classes, dando o mesmo peso a benigno e maligno.
        
        • Weighted avg: média que considera quantos casos havia em cada classe; 
            aqui, dá mais peso aos 54 casos benignos do que aos 32 malignos.
        """
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

        importance = permutation_importance(
            self.final_model,
            self.modeling.X_test,
            self.modeling.y_test,
            scoring="recall",
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
            "Limitação: estes resultados vêm de uma única divisão do dataset e não "
            "demonstram segurança clínica. O modelo é apenas apoio à análise; não "
            "substitui exames, avaliação profissional ou a decisão do médico."
        )
