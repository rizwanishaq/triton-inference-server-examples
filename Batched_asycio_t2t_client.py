from uuid import uuid4
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
import random
import numpy as np
import asyncio
import time

# Constants for Model Names and URLs
DEFAULT_MODEL_NAME = "t2t_lang"
DEFAULT_TRITON_URL = "0.0.0.0:8001"

# List of sample sentences in different languages
sample_sentences = [
    "Ich lebe in Berlin, der Hauptstadt von Deutschland.",
    "Je vis à Paris, la capitale de la France.",
    "Vivo a Roma, la capitale dell'Italia.",
    "Vivo en Madrid, la capital de España.",
    "Eu moro em Lisboa, a capital de Portugal.",
    "मैं भारत में रहता हूँ।",
    "مرحبا بك في العالم.",
    "من ہندوستان میں رہتا ہوں۔",
    "I live in New York, the city that never sleeps.",
    "我住在北京，这是中国的首都。",
    "Tokyo is a bustling city in Japan.",
    "Seoul is the capital of South Korea.",
    "J'habite à Montréal, une belle ville au Canada.",
    "Я живу в Москве, городе-герое России.",
    "Eu moro em São Paulo, a maior cidade do Brasil.",
    "أنا أعيش في القاهرة، عاصمة مصر.",
    "من در تهران زندگی می‌کنم، پایتخت ایران.",
    "मैं दिल्ली में रहता हूँ, भारत की राजधानी।",
    "من در تهران زندگی می‌کنم، پایتخت ایران.",
    "Je vis à Montréal, une ville multiculturelle.",
    "Eu moro em Lisboa, a cidade das sete colinas.",
    "Я живу в Санкт-Петербурге, культурной столице России.",
    "أنا أعيش في جدة، مدينة الأمان والأمانة في المملكة العربية السعودية.",
    "من در تهران زندگی می‌کنم، پایتخت ایران.",
    "मैं भारत में रहता हूँ।",
    "مرحبا بك في العالم.",
    "من ہندوستان میں رہتا ہوں۔",
    "I live in New York, the city that never sleeps.",
    "我住在北京，这是中国的首都。",
    "Tokyo is a bustling city in Japan.",
    "Seoul is the capital of South Korea.",
    "J'habite à Montréal, une belle ville au Canada.",
    "Я живу в Москве, городе-герое России.",
    "Eu moro em São Paulo, a maior cidade do Brasil.",
    "أنا أعيش في القاهرة، عاصمة مصر.",
    "من در تهران زندگی می‌کنم، پایتخت ایران."
]


class TritonClient:
    """
    A client for communicating with Triton Inference Server.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, triton_url: str = DEFAULT_TRITON_URL) -> None:
        """
        Initializes the Triton client.

        Args:
            model_name (str, optional): Name of the model. Defaults to "t2t_lang".
            triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".
        """
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False)
        self.model_name = model_name

    def infer(self, text_input: str) -> (str, float):
        """
        Calls the Triton client to process a text input.

        Args:
            text_input (str): The text input for processing.

        Returns:
            str: Detected Language.
            float: Confidence Score.
        """
        input_txt = np.array([[text_input]], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "input_txt",
                input_txt.shape,
                np_to_triton_dtype(input_txt.dtype)
            ),
        ]
        outputs = [
            grpcclient.InferRequestedOutput("language"),
            grpcclient.InferRequestedOutput("score"),
        ]
        inputs[0].set_data_from_numpy(input_txt)

        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=str(uuid4()),
            outputs=outputs,
        )

        language = response.as_numpy("language")[0].decode()
        score = response.as_numpy("score")[0]

        return language, score


async def client_async(client: "TritonClient", text_input: str) -> None:
    """
    Executes Triton client for a specific text input in an asynchronous manner.

    Args:
        client (TritonClient): The Triton client object.
        text_input (str): The text input for processing.
    """
    language, score = await asyncio.to_thread(client.infer, text_input)
    print(
        f"Input Text: {text_input}\nDetected Language: {language}\nConfidence Score: {score}\n")


async def main():
    num_clients = 30
    clients = [TritonClient() for _ in range(num_clients)]
    text_inputs = random.choices(sample_sentences, k=num_clients)

    # Using asyncio
    start_time_asyncio = time.time()

    await asyncio.gather(*(client_async(client, text_input) for client, text_input in zip(clients, text_inputs)))

    end_time_asyncio = time.time()
    print(
        f"Time taken using asyncio: {end_time_asyncio - start_time_asyncio} seconds")

if __name__ == "__main__":
    asyncio.run(main())
