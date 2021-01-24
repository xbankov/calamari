FROM tensorflow/tensorflow:2.3.0-gpu
RUN apt-get update && apt-get install git -y && git clone https://github.com/xbankov/calamari.git && cd calamari && pip install .

