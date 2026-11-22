from src.cancer_de_mama.data_exploration.data_exploration import DataExploration
from src.cancer_de_mama.model_training_and_evaluation.model_training_and_evaluation import ModelTrainingAndEvaluation
from src.cancer_de_mama.modeling.modeling import Modeling
from src.cancer_de_mama.preprocessing.preprocessing import Preprocessing


class BreastCancer:

    def __init__(self):
        """Carregando o DataSet..."""
        self.data_frame = None

    def execute(self):
        print("\n====================================================================================")
        print("\t\t\t\t\t\t\t\tCÂNCER DE MAMA")
        print("====================================================================================")

        """ Exploração de dados """
        self.data_frame = DataExploration().execute()

        """ Pré-processamento de dados """
        self.data_frame = Preprocessing(self.data_frame).execute()

        """ Modelagem """
        modeling = Modeling(self.data_frame).execute()

        """ Treinamento e Avaliação do modelo """
        ModelTrainingAndEvaluation(modeling).execute()
