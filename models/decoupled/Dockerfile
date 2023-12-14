FROM nvcr.io/nvidia/tritonserver:23.08-py3
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get --yes install libsndfile1
RUN pip install --upgrade pip
RUN pip install boto
RUN pip install SoundFile
RUN pip install librosa
RUN pip install opencv-python
RUN pip install mediapipe
RUN pip install torch
RUN pip install torchvision
RUN pip install face-detection
RUN pip install face-alignment
RUN pip install imageio
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install uuid
RUN pip install imageio[ffmpeg]
RUN pip install PyYAML
RUN pip install safetensors
RUN pip install numba
RUN pip install resampy
RUN pip install scipy
RUN pip install tqdm
RUN pip install basicsr==1.4.2
RUN pip install facexlib==0.3.0
RUN pip install gfpgan
RUN pip install av
RUN pip install kornia==0.6.8
RUN pip install yacs==0.1.8
RUN pip install pydub==0.25.1
RUN pip install fasttext
