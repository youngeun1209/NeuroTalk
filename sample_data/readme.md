## Sample data
Since the dataset contains human-derived biosignal, we only open a small amount of sample dataset to reproduce and run our code.

### Test data
Five test data were provided, including the words 'yes', 'hello', 'help me', 'water', 'pain'.

### Infer
To evaluate the trained model for spoken EEG on an example data, run:
```eval
python eval.py pretrained_model/SpokenEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task SpokenEEG_vec --batch_size 5
```
To evaluate the trained model for Imagined EEG on an example data, run:
```eval
python eval.py pretrained_model/ImaginedEEG/ pretrained_model/UNIVERSAL_V1/g_02500000 --task ImaginedEEG_vec --batch_size 5
```

> You can infer the model with this sample data based on pretrained model
