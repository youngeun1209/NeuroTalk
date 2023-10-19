# NeuroTalk: Voice Reconstruction from Brain Signals
This repository is the official implementation of Towards Voice Reconstruction from EEG during Imagined Speech.
This paper is published in AAAI 2023.

## Citation
Y.-E. Lee, S.-H. Lee, S.-H Kim, and S.-W. Lee, "Towards Voice Reconstruction from EEG during Imagined Speech," AAAI Conference on Artificial Intelligence (AAAI), 2023.

## Requirements
All algorithm are developed in Python 3.8.

To install requirements:

```setup
pip install -r requirements.txt
```

Vector Embedding needs BBCI toolbox in MATLAB.

You can download here:

https://github.com/bbci/bbci_public

## Training
To train the model for spoken EEG in the paper, run this command:
```train
python train.py --vocoder_pre pretrained_model/UNIVERSAL_V1/g_02500000 --task SpokenEEG_vec --batch_size 20 --pretrain False --prefreeze False
```
To train the model for Imagined EEG with pretrained model of spoken EEG in the paper, run this command:
```train
python train.py --vocoder_pre pretrained_model/UNIVERSAL_V1/g_02500000 --trained_model pretrained_model/SpokenEEG/ --task ImaginedEEG_vec --batch_size 20 --pretrain True --prefreeze True
```

## Evaluation
To evaluate the trained model for spoken EEG on an example data, run:
```eval
python eval.py --trained_model pretrained_model/SpokenEEG/ --vocoder_pre pretrained_model/UNIVERSAL_V1/g_02500000 --task SpokenEEG_vec --batch_size 5
```
To evaluate the trained model for Imagined EEG on an example data, run:
```eval
python eval.py --trained_model pretrained_model/ImaginedEEG/ --vocoder_pre pretrained_model/UNIVERSAL_V1/g_02500000 --task ImaginedEEG_vec --batch_size 5
```

## Demo page
- [Demo page](https://neurotalk.github.io/demo)


## Pre-trained Models
You can download pretrained models here:
- [Pretrained model](https://drive.google.com/drive/folders/1x6GNHzAQkqL5eQmIcPTjVPb9D5dtx02W?usp=sharing) trained on participant 1


## Citation
Y.-E. Lee, S.-H. Lee, S.-H Kim, and S.-W. Lee, "Towards Voice Reconstruction from EEG during Imagined Speech," AAAI Conference on Artificial Intelligence (AAAI), 2023.


## Contributing
- We propose a generative model based on multi-receptive residual modules with recurrent neural networks that can extract frequency characteristics and sequential information from neural signals, to generate speech from non-invasive brain signals.

- The fundamental constraint of the imagined speech-based BTS system lacking the ground truth voice have been addressed with the domain adaptation method to link the imagined speech EEG, spoken speech EEG, and the spoken speech audio.

- Unseen words were able to be reconstructed from the pre-trained model by using character-level loss to adapt various phonemes. This implies that the model could learn the phoneme level information from the brain signal, which displays the potential of robust speech generation by training only several words or phrases.

