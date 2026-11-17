from src.utils.Constants import Constants
from src.cancer_de_mama.preprocessing.plot_correlation_matrix import MatrixCorrelation


class Preprocessing:

    """
    Pré-processamento de dados:
        ● Realize a limpeza dos dados, tratando valores ausentes e inconsistentes (se necessário);
        ● Pipeline de pré-processamento de dados em Python.
        ● Converta variáveis categóricas e numéricas em formatos adequados para modelagem.
        ● Realize a análise de correlação.
    """

    def __init__(self, data_frame):
        self.feature_columns = None
        self.df_feature_columns = None
        self.data_frame = data_frame

    def execute(self):
        print("\n--------- Pré-processamento de dados -----------------------------------------------")
        self.data_frame = self.clean_columns_empty()
        self.data_frame = self.null_and_duplicate_values()
        self.data_frame = self.target_to_integer()
        self.data_frame = self.df_columns_without_id()

        print("\nMatriz de Correlação")
        MatrixCorrelation(self.data_frame).execute()

        return self.data_frame


    def clean_columns_empty(self):
        print("\nRemovendo colunas completamente vazias do dataset...")
        self.data_frame = self.data_frame.dropna(axis=1, how="all")
        return self.data_frame

    def null_and_duplicate_values(self):
        print("\nValidando campos ausentes por coluna, duplicados, id´s duplicados")
        missing_values_per_column = self.data_frame.isnull().sum()
        total_missing_values_per_column = missing_values_per_column.sum()
        duplicate_rows_count = self.data_frame.duplicated().sum()
        duplicate_id_count = self.data_frame["id"].duplicated().sum()
        missing_values = total_missing_values_per_column + duplicate_rows_count + duplicate_id_count

        if missing_values > 0:
            print("ATENÇÃO: Foram encontrados problemas de qualidade nos dados!")
            print("\nValores ausentes por coluna:")
            print(missing_values_per_column)

            print("\nTotal de valores ausentes:")
            print(total_missing_values_per_column)

            print("\nQuantidade de registros duplicados:")
            print(duplicate_rows_count)

            print("\nQuantidade de IDs duplicados:")
            print(duplicate_id_count)
        else:
            print("Nenhum problema de qualidade encontrado nos dados.\n")

        return self.data_frame

    def set_df_feature_columns(self):
        self.feature_columns = [
            column for column in self.data_frame.columns
            if column not in [Constants.DIAGNOSIS_COLUMN, Constants.ID_COLUMN]
        ]
        self.df_feature_columns = self.data_frame[self.feature_columns]
        return self.df_feature_columns

    def target_to_integer(self):
        self.data_frame[Constants.DIAGNOSIS_COLUMN] = (
            self.data_frame[Constants.DIAGNOSIS_COLUMN].map({
                Constants.DIAGNOSIS_BENIGN: Constants.BENIGN_VALUE,
                Constants.DIAGNOSIS_MALIGNANT: Constants.MALIGNANT_VALUE
            })
        )
        return self.data_frame


    def df_columns_without_target(self):
        columns = [
            column for column in self.data_frame.columns
            if column not in [Constants.DIAGNOSIS_COLUMN]
        ]
        self.data_frame = self.data_frame[columns]
        return self.data_frame

    def df_columns_without_id(self):
        columns = [
            column for column in self.data_frame.columns
            if column not in [Constants.ID_COLUMN]
        ]
        self.data_frame = self.data_frame[columns]
        return self.data_frame
