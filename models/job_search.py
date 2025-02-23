import json
from datasets import load_dataset

class JobSearch:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)

    def get_job_descriptions(self):
        job_descriptions = []
        for example in self.dataset["train"]:
            job_descriptions.append(example["job_description_text"])  # Use the correct key
        return job_descriptions

    def save_job_descriptions(self, output_path):
        job_descriptions = self.get_job_descriptions()
        with open(output_path, "w") as f:
            json.dump(job_descriptions, f)