import os
from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher
import shutil

class MyLauncher(SageMakerCoachPresetLauncher):

    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return 'preset-portfolio-management-clippedppo'

    def map_hyperparameter(self, name, value):
        """Here we configure some shortcut names for hyperparameters that we expect to use frequently.
        Essentially anything in the preset file can be overridden through a hyperparameter with a name
        like "rl.agent_params.algorithm.etc".
        """
        # maps from alias (key) to fully qualified coach parameter (value)
        mapping = {
            "discount": "rl.agent_params.algorithm.discount",
            "improve_steps": "rl.improve_steps:TrainingSteps",
            "training_epochs": "rl.agent_params.algorithm.optimization_epochs",
            "evaluation_episodes": "rl.evaluation_steps:EnvironmentEpisodes"}
        if name in mapping:
            self.apply_hyperparameter(mapping[name], value)
        else:
            super().map_hyperparameter(name, value)


if __name__ == '__main__':
    # Signal the environment that this is a training session and hide the test data
    with open(os.path.join(os.path.dirname(__file__), 'session-type.txt'), 'w') as f:
        f.write('train')
    
    # Launch training.
    MyLauncher.train_main()
