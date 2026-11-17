from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.Constants import Constants


class PlotDiagnosisDistribution:
    def __init__(self, data_frame):
        self.output_dir = Path("results/graphics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = data_frame
        self.target_column = Constants.DIAGNOSIS_COLUMN

    def execute(self):
        """Gera um gráfico com a quantidade de casos benignos e malignos."""
        print("Arquivo gerado: distribuicao_diagnosticos.png")

        plt.figure(figsize=(8, 5))

        ax = sns.countplot(
            data=self.df,
            x=self.target_column,
            hue=self.target_column,
            order=[Constants.DIAGNOSIS_BENIGN, Constants.DIAGNOSIS_MALIGNANT],
            palette={
                Constants.DIAGNOSIS_BENIGN: "#2E86AB",
                Constants.DIAGNOSIS_MALIGNANT: "#FF8C00",
            },
            legend=False
        )

        ax.set_title("Distribuição dos Diagnósticos")
        ax.set_xlabel("Diagnóstico")
        ax.set_ylabel("Quantidade")

        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "distribuicao_diagnosticos.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
