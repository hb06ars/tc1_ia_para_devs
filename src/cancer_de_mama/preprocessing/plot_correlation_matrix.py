from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class MatrixCorrelation:
    def __init__(self, data_frame):
        self.feature_columns = data_frame
        self.output_dir = Path("results/graphics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = data_frame
        sns.set_theme(style="whitegrid")

    def execute(self):
        """Exibe e salva correlações entre medidas; valores próximos de 1 indicam relação positiva.
        Importante para identificar quais medidas estão mais relacionadas entre si e com o diagnóstico.
        """
        print("O quanto uma variável está relacionada a outra, o quanto um aumento no valor de uma variável "
              "resulta em um aumento ou diminuição no valor da outra.")
        print("Arquivo gerado: matriz_correlacao.png")

        correlation = self.df.corr()

        plt.figure(figsize=(22, 18))

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=False,
            square=True
        )

        plt.title("Matriz de Correlação das Características")

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "matriz_correlacao.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        correlation.to_csv(
            self.output_dir / "matriz_correlacao.csv"
        )
