# pytorch-modulation-recognition
PyTorch Implementation of various Modulation Recognition Networks benchmarked on the RadioML2016 Dataset

## Install Requirements

```bash
# Base requirements
pip3 install -r requirements.txt

# Required to use train.py
pip3 install poutyne

# Required to run the demo web app
pip3 install streamlit

```

## Train

```bash
# Train VTCNN2
python train.py --model vtcnn --epochs 25 --batch_size 512 --split 0.8

# Train MRResNet
python train.py --model mrresnet --epochs 25 --batch_size 512 --split 0.8

```

### Results

After training Tensorboard logs will be located in the logs directory and the models in the models directory. Run the below command to start Tensorboard and point your browser to localhost:6006.

```bash
tensorboard --logdir=logs

```

### Demo

To run the web app for visualizing the In-Phase (I) and Quadrature (Q) channels of signals at various modulations and signal-to-noise ratios, use the following command.

```bash
streamlit run app.py

```