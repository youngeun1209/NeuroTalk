# NeuroTalk: Voice Reconstruction from Brain Signals
This repository is the official implementation of Towards Voice Reconstruction from EEG during Imagined Speech

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the model for spoken EEG in the paper, run this command:
```train
python train.py pretrained_model/SpokenEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task SpokenEEG_vec --batch_size 20 --pretrain False --prefreeze False
```
To train the model for Imagined EEG with pretrained model of spoken EEG in the paper, run this command:
```train
python train.py pretrained_model/SpokenEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task ImaginedEEG_vec --batch_size 20 --pretrain True --prefreeze True
```
>📋 the arguments of models


## Evaluation
To evaluate the trained model for spoken EEG on an example data, run:
```eval
python eval.py pretrained_model/SpokenEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task SpokenEEG_vec --batch_size 5
```
To evaluate the trained model for Imagined EEG on an example data, run:
```eval
python eval.py pretrained_model/ImaginedEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task ImaginedEEG_vec --batch_size 5
```

## Demo page
- [Demo page](https://neurotalk.github.io/demo/neurotalk.html)

## Citation
Y.-E. Lee, S.-H. Lee, S.-H Kim, and S.-W. Lee, "Towards Voice Reconstruction from EEG during Imagined Speech," AAAI Conference on Artificial Intelligence (AAAI), 2023.


## Contributing
- We propose a generative model based on multi-receptive residual modules with recurrent neural networks that can extract frequency characteristics and sequential information from neural signals, to generate speech from non-invasive brain signals.

- The fundamental constraint of the imagined speech-based BTS system lacking the ground truth voice have been addressed with the domain adaptation method to link the imagined speech EEG, spoken speech EEG, and the spoken speech audio.

- Unseen words were able to be reconstructed from the pre-trained model by using character-level loss to adapt various phonemes. This implies that the model could learn the phoneme level information from the brain signal, which displays the potential of robust speech generation by training only several words or phrases.


<!--
**NeuroTalk/NeuroTalk** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
