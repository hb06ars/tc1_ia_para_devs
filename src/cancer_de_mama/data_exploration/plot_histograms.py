from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.Constants import Constants


class Histograms:
    def __init__(self, data_frame):
        self.output_dir = Path("results/graphics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = data_frame

        # Identifica as variáveis que não são características preditoras
        self.target_column = Constants.DIAGNOSIS_COLUMN
        self.id_column = Constants.ID_COLUMN
        sns.set_theme(style="whitegrid")

    def execute(self):
        """ Gera histogramas para comparar a distribuição de cada medida entre diagnósticos.
            Eixo X: valor da característica, exemplo: radius_mean.
            Eixo Y: quantidade de amostras (frequência) que apresentam valores naquela faixa.
        """
        print("Histogramas para cada coluna, mostrando a frequência entre os diagnósticos Maligno (M) e Benigno (B).")
        print("Arquivo gerado: distribuicao_caracteristicas.png")

        feature_columns = [
            column
            for column in self.df.columns
            if column not in (self.target_column, self.id_column)
            and self.df[column].notna().any()
        ]
        number_of_features = len(feature_columns)
        ncols = 5
        nrows = max(1, (number_of_features + ncols - 1) // ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(20, 22)
        )

        axes = axes.flatten()

        for index, column in enumerate(feature_columns):
            sns.histplot(
                data=self.df,
                x=column,
                hue=self.target_column,
                palette={
                    Constants.DIAGNOSIS_BENIGN: "#2E86AB",
                    Constants.DIAGNOSIS_MALIGNANT: "#FF8C00",
                },
                kde=True,
                ax=axes[index],
                element="step"
            )

            axes[index].set_title(column, fontsize=9)
            axes[index].set_xlabel("Valor")
            axes[index].set_ylabel("Frequência")

        # Remove espaços vazios, caso existam
        for index in range(number_of_features, len(axes)):
            fig.delaxes(axes[index])

        fig.suptitle(
            "Distribuição das Características por Diagnóstico",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "distribuicao_caracteristicas.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()