from polyaxon_client.tracking import get_outputs_path
from pathlib import Path


def get_polyaxon_resume_file(models_dir_name):
    output_path = Path(get_outputs_path())

    models_path = output_path / models_dir_name
    experiments_path = [x for x in models_path.iterdir() if x.is_dir()][0]
    experiments = [x for x in experiments_path.iterdir() if x.is_dir()]
    # Take the latest experiment
    experiments.sort(reverse=True)
    resume_from = experiments[0]

    checkpoint_file = [x for x in resume_from.iterdir(
    ) if x.suffixes[0] == ".pth" and x.name != "model_best.pth"][0]
    return checkpoint_file
