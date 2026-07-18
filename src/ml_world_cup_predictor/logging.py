"""A module that will capture detail about a single model run and store for comparison later"""
import uuid
import datetime as dt
from typing import Annotated
from pydantic import BaseModel, Field, model_validator

from ml_world_cup_predictor.config import DATA_PATH

UnitInterval = Annotated[float, Field(ge=0, le=1)]

class PerClassMetrics(BaseModel):
    recall: dict[str, UnitInterval]
    precision: dict[str, UnitInterval]

    # Check to ensure both sets of metrics have the same class labels, else raise ValueError
    @model_validator(mode='after')
    def check_matching_labels(self):
        if self.recall.keys() != self.precision.keys():
            raise ValueError(f'recall and precision must have the same class label: '
                             f'{sorted(self.recall.keys())} != {sorted(self.precision.keys())}')


        return self
class ModelMetrics(BaseModel):
    accuracy:  float = Field(ge=0, le=1)
    class_metrics: PerClassMetrics

class RunRecord(BaseModel):
    run_id: str
    run_datetime: str
    run_time: str
    metrics: ModelMetrics


def write_run(run_time:str,metrics:ModelMetrics)-> None:
    """Writes the logs for a model run to a json file with UUID key"""
    # Ensure path exists ahead of write
    write_path = DATA_PATH.run
    write_path.mkdir(exist_ok=True, parents=True)

    run = RunRecord(
    run_id          = str(uuid.uuid4()), # Creating a new run_id for each model run
    run_datetime    = dt.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), # Date and Time to second of model run
    run_time        = run_time,
    metrics         = metrics
    )

    with open(f'{write_path}/{run.run_id}.json','w') as f:
        f.write(run.model_dump_json(indent=2))

