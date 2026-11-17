from pathlib import Path

import pandas as pd


class DataSetLoad:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {self.file_path}")

        self.df = pd.read_csv(self.file_path)

        print("\nDataset carregado com sucesso...\n")

        return self.df
