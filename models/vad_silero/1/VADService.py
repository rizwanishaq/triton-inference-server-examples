import os
import numpy as np
import torch
from utils import VADIterator


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) class.

    Args:
        sampling_rate (int, optional): Sampling rate of the audio (default: 16000).
        model_path (str, optional): Path to the pre-trained VAD model (default: "pretrainedmodel/silero_vad.jit").
    """
    FRAME_SIZE: int = 320     # 20ms with sampling rate 16000
    BUFFER_SIZE: int = 512    # Buffer size for processing the audio for the model

    def __init__(self, sampling_rate: int = 16000, model_path: str = "pretrainedmodel/silero_vad.jit"):
        """
        Initialize the Voice Activity Detector.

        Args:
            sampling_rate (int, optional): Sampling rate of the audio (default: 16000).
            model_path (str, optional): Path to the pre-trained VAD model (default: "pretrainedmodel/silero_vad.jit").
        """
        self.sampling_rate: int = sampling_rate
        self.model_path: str = os.path.join(
            "/models/vad_silero/1/", model_path)
        self.model: torch.jit.ScriptModule = torch.jit.load(
            self.model_path, map_location=torch.device('cpu'))
        self.vad_iterator: VADIterator = VADIterator(
            model=self.model,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=100,
            speech_pad_ms=30,  # Adjusted to 20ms for VAD processing
            threshold=0.5  # Adjusted threshold for your use case
        )

        self.input_audio_buffer: np.ndarray = np.zeros(
            (self.BUFFER_SIZE,), dtype=np.int16)
        self.output_audio_buffer: np.ndarray = np.array([], np.int16)
        self.full_audio_buffer: np.ndarray = np.array([], np.int16)
        self.start: int = 0
        self.end: int = 0
        self.speech_flag: bool = False
        self.chunk_size: int = 512  # Number of samples per chunk for processing

    def convert_to_float32(self, sound: np.ndarray) -> np.ndarray:
        """
        Convert audio samples to float32.

        Args:
            sound (np.ndarray): Input audio samples.

        Returns:
            np.ndarray: Audio samples in float32 format.
        """
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        return sound

    def reset(self) -> None:
        """
        Reset the VAD state and save clean audio as WAV file.
        """
        self.vad_iterator.reset_states()
        del self.model

    def buffer_audio(self, chunk: np.ndarray) -> None:
        """
        Buffer incoming audio for processing.

        Args:
            chunk (np.ndarray): Audio chunk as an ndarray.
        """
        self.full_audio_buffer = np.concatenate(
            [self.full_audio_buffer, chunk])
        self.input_audio_buffer = np.roll(
            self.input_audio_buffer, -self.FRAME_SIZE)
        self.input_audio_buffer[-self.FRAME_SIZE:] = chunk

    def get_clean_audio_chunk(self) -> np.ndarray:
        """
        Get a chunk of clean audio samples.

        Returns:
            np.ndarray: Chunk of audio samples or None.
        """
        if len(self.output_audio_buffer) >= self.FRAME_SIZE:
            chunk = self.output_audio_buffer[:self.FRAME_SIZE]
            self.output_audio_buffer = self.output_audio_buffer[self.FRAME_SIZE:]
            return chunk
        return None

    def process_speech_dict(self, speech_dict: dict) -> None:
        """
        Process the speech dictionary obtained from VAD.

        Args:
            speech_dict (dict): Dictionary containing speech information.
        """
        if self.speech_flag and speech_dict is None:
            end = self.start + min(self.FRAME_SIZE,
                                   len(self.full_audio_buffer))
            self.output_audio_buffer = np.concatenate(
                [self.output_audio_buffer, self.full_audio_buffer[self.start:end]])
            self.start = end

        if speech_dict is not None:
            if speech_dict.get('start', False):
                self.start = speech_dict.get('start')
                self.speech_flag = True
            else:
                self.end = speech_dict.get('end')
                self.speech_flag = False

                if self.end > self.start:
                    self.output_audio_buffer = np.concatenate(
                        [self.output_audio_buffer, self.full_audio_buffer[self.start:self.end]])

        if speech_dict is None and not self.speech_flag:
            end = self.end + min(self.FRAME_SIZE, len(self.full_audio_buffer))

            self.output_audio_buffer = np.concatenate(
                [self.output_audio_buffer, np.zeros(end - self.end, dtype=np.int16)])
            self.end = end

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process an audio chunk and return the VAD result.

        Args:
            chunk (np.ndarray): Audio chunk as an ndarray.

        Returns:
            np.ndarray: Audio chunk of size self.FRAME_SIZE or None.
        """
        self.buffer_audio(chunk)

        chunk_for_processing = self.convert_to_float32(self.input_audio_buffer)

        speech_dict = self.vad_iterator(
            chunk_for_processing, return_seconds=False, window_size_samples=self.FRAME_SIZE)

        self.process_speech_dict(speech_dict)

        return self.get_clean_audio_chunk()
