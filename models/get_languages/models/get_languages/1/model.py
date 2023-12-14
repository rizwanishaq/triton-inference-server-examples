import json
import triton_python_backend_utils as pb_utils
from typing import List, Dict, Optional
import numpy as np


class TritonPythonModel:

    def initialize(self, args: Dict[str, str], languages_file: str = '/models/get_languages/1/languages.json') -> None:
        """
        Initialize the TritonPythonModel.

        Parameters:
        - args (Dict[str, str]): Model configuration arguments.
        - languages_file (str, optional): Path to the file storing the list of languages.
        """
        self.model_config: dict = json.loads(args['model_config'])
        self.languages_file = languages_file
        self.languages = self.load_languages()

        # Get transcription configuration
        languages_config = pb_utils.get_output_config_by_name(
            self.model_config, "languages")

        # Convert Triton types to numpy types
        self.languages_dtype = pb_utils.triton_string_to_numpy(
            languages_config['data_type'])
        self.logger = pb_utils.Logger

    def load_languages(self) -> List[str]:
        """
        Load the list of languages from the specified file.

        Returns:
        - List[str]: List of languages.
        """
        try:
            with open(self.languages_file, 'r') as file:
                languages = json.load(file)
            return languages
        except FileNotFoundError:
            print(f'Languages file not found. Creating a new one.')
            return []

    def save_languages(self) -> None:
        """
        Save the current list of languages to the specified file.
        """
        with open(self.languages_file, 'w') as file:
            json.dump(self.languages, file, indent=2)

    def add_language(self, new_language: str) -> None:
        """
        Add a new language to the list and save the updated list.

        Parameters:
        - new_language (str): New language to be added.
        """
        if new_language not in self.languages:
            self.languages.append(new_language)
            self.save_languages()

    def execute(self, requests: List):
        """
        Execute the model.

        Parameters:
        - requests (List): List of inference requests.

        Returns:
        - List[pb_utils.InferenceResponse]: List of inference responses.
        """
        responses = []

        for request in requests:
            parameters = json.loads(request.parameters())

            self.logger.log(f'{parameters}', self.logger.INFO)

            new_language = parameters.get('new_language', None)
            if new_language:
                self.add_language(new_language)

            # Append the response to the list
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    'languages',
                    np.array([json.dumps(self.languages)],
                             dtype=self.languages_dtype),
                )
            ]))

        return responses

    def finalize(self) -> None:
        """
        Finalize the model.
        """
        print('ASR Service cleaning up...')
