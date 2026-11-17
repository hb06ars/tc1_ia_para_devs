from pathlib import Path

from src.cancer_de_mama.data_exploration.load_dataset import DataSetLoad
from src.cancer_de_mama.data_exploration.plot_diagnosis_distribution import PlotDiagnosisDistribution
from src.cancer_de_mama.data_exploration.plot_histograms import Histograms


class DataExploration:
    """
    Exploração de dados:
        ● Carregue a base de dados e explore as suas características;
        ● Analise estatísticas descritivas e visualize distribuições relevantes, discutindo os resultados;
    """

    def __init__(self):
        self.data_frame = None
        self.output_dir = Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("\n--------- Exploração de dados ------------------------------------------------------")

    def execute(self):
        self.data_frame = DataSetLoad(file_path="data/breast_cancer.csv").load()

        """
            - Carregamos o arquivo breast_cancer.csv dentro da pasta data.
            - Vejo que abaixo existe uma coluna id, que deve ser o id do paciente, para identificá-lo.
            - Tem também a coluna "diagnosis" que é string, que possui valores M e B, logo entendo que seja a variável target, conforme execuções abaixo.
            - O restante é tudo float.
            - E vejo de início que existem colunas nulas, por conta de uma vírgula no final das colunas do csv.
              A limpeza dessas colunas ocorre na etapa de pré-processamento.
        """

        print("Dimensões do DataFrame:")
        print("Linhas: ", self.data_frame.shape[0])
        print("Colunas: ", self.data_frame.shape[1])

        print("\nCaracterísticas das colunas:")
        print(self.data_frame.info())

        print("\nCabeçalho 5 primeiros itens:")
        print(self.data_frame.head())

        statistics = self.data_frame.drop(columns=["id"], errors="ignore").describe()
        print("Média, Mín, Max...:")
        print(statistics)
        statistics.to_csv(self.output_dir / "estatisticas_descritivas.csv")

        print("\nTotal Benignos x Malignos: ")
        print(self.data_frame["diagnosis"].value_counts())

        print("\nPercentual Benignos x Malignos: ")
        print(self.data_frame["diagnosis"].value_counts(normalize=True) * 100)

        print("\nSalvando gráfico Benignos X Malignos")
        PlotDiagnosisDistribution(self.data_frame).execute()

        print("\nSalvando gráfico Histogramas")
        Histograms(self.data_frame).execute()

        return self.data_frame
