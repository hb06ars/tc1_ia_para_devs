from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Modeling:
    """
    Modelagem:
        ● Crie modelos preditivos de classificação utilizando duas ou mais técnicas à sua escolha (por exemplo: Regressão logística, Árvore de Decisão, KNN etc);
    	● Separação clara entre treino, validação e teste.
    """

    TARGET_COLUMN = "diagnosis"
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    RANDOM_STATE = 42

    def __init__(self, data_frame=None):
        self.data_frame = data_frame
        self.X_train = None
        self.X_validation = None
        self.X_test = None
        self.y_train = None
        self.y_validation = None
        self.y_test = None
        self.models = {}

    def execute(self, data_frame=None):
        print("\n--------- Modelagem ----------------------------------------------------------------\n")

        """ Jogando valores das features (colunas) e id (target) para X e y: X contém as features, e y contém a coluna target. """
        X, y = self.get_features_and_target(data_frame)

        """ SEPARANDO: 70% TREINO, 15% TESTE e 15% VALIDAÇÃO """
        self.get_train_validation_test(X, y)

        """ Define os pipelines de pré-processamento e os modelos de classificação: 
            (Regressão Logística, SVM e Árvore de Decisão).
        """
        self.save_models_and_pipelines()

        """ Printa os valores e porcentagens de Treino, Validação e Teste:"""
        self.print_info_validations()

        return self

    def print_info_validations(self):
        train_size = 1 - self.TEST_SIZE - self.VALIDATION_SIZE
        print(f"Treino ({train_size:.0%}), Validação ({self.VALIDATION_SIZE:.0%}), "f"Teste ({self.TEST_SIZE:.0%})")
        print(f"Treino: {len(self.X_train)} registros")
        print(f"Validação: {len(self.X_validation)} registros")
        print(f"Teste: {len(self.X_test)} registros")
        print("Modelos definidos: " + ", ".join(self.models))

    def save_models_and_pipelines(self):
        # Regressão Logística: modelo inicial para o problema binário (benigno ou maligno)
        # Serve como referência para comparar os demais modelos e pode estimar
        # a probabilidade de cada classe. A padronização coloca as medidas
        # em escalas semelhantes, o que ajuda o modelo a aprender com elas.
        self.models = {
            "Regressão Logística": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=1000, random_state=self.RANDOM_STATE)),
                ]
            ),

            # SVM: pode funcionar bem com um conjunto pequeno de registros e
            # muitas características, como neste dataset. O kernel RBF permite
            # identificar padrões não lineares entre as medidas e o diagnóstico.
            # A padronização é importante porque o SVM é sensível à escala das medidas.
            "SVM": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        CalibratedClassifierCV(
                            estimator=SVC(kernel="rbf", random_state=self.RANDOM_STATE),
                            method="sigmoid",
                            cv=5,
                            ensemble=False,
                        ),
                    ),
                ]
            ),

            # Árvore de Decisão: divide os dados em regras simples, como verificar
            # se uma medida está acima ou abaixo de determinado valor. Isso pode
            # facilitar a compreensão de como o modelo chega a uma classificação.
            # Não precisa de padronização, mas uma árvore muito profunda pode
            # memorizar os dados de treino e ter dificuldade com dados novos (viciar).
            "Árvore de Decisão": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", DecisionTreeClassifier(random_state=self.RANDOM_STATE)),
                ]
            ),
        }

    def get_train_validation_test(self, X, y):
        """
                SEPARANDO: 70% TREINO, 15% TESTE e 15% VALIDAÇÃO.
                Separa 15% dos dados originais para o teste.
                Sobram 85% em treino_e_validação (X_train_validation e y_train_validation).
            """
        X_train_validation, self.X_test, y_train_validation, self.y_test = train_test_split(
            X,
            y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=y,
        )

        """ A validação deve representar 15% dos dados originais.
            Por isso, test_size nesta segunda separação é 15% / 85% do conjunto restante.
            Os registros restantes formam aproximadamente 70% para treinamento.
        """
        validation_fraction = self.VALIDATION_SIZE / (1 - self.TEST_SIZE)
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=validation_fraction,
            random_state=self.RANDOM_STATE,
            stratify=y_train_validation,
        )

    def get_features_and_target(self, data_frame) -> tuple[Any, Any]:
        data_frame = data_frame if data_frame is not None else self.data_frame
        if data_frame is None:
            raise ValueError("O DataFrame pré-processado deve ser informado para a modelagem.")
        if self.TARGET_COLUMN not in data_frame.columns:
            raise ValueError(f"A coluna target '{self.TARGET_COLUMN}' não foi encontrada.")

        X = data_frame.drop(columns=[self.TARGET_COLUMN, "id"], errors="ignore")
        y = data_frame[self.TARGET_COLUMN]

        if X.empty:
            raise ValueError("Nenhuma variável preditora foi encontrada.")
        if y.isna().any():
            raise ValueError("A coluna target contém valores ausentes.")
        if y.nunique() < 2:
            raise ValueError("A coluna target precisa conter pelo menos duas classes.")

        return X, y
