from speechbrain.pretrained import EncoderClassifier
import torchaudio
import pandas as pd
import os

classifier = EncoderClassifier.from_hparams(source="/home/armine/Codes/star_rail_ai/speaker_classification/results/speaker_id/1987/save/CKPT+2023-05-08+22-45-26+00",
                                            hparams_file='hparams_inference.yaml',
                                            savedir="/home/armine/Codes/star_rail_ai/speaker_classification/results/speaker_id/1987/save/CKPT+2023-05-08+22-45-26+00")

DATA_PATH = "/home/armine/Dataset/deciphered_wav"
csv_data = pd.read_csv('../configs/datasheet.csv')
csv_range = (384, 1133)
data_json = {}
for i in range(*csv_range):
    data_row = csv_data.loc[i]
    audio_path = os.path.join(DATA_PATH, data_row['Path'])

    signal, fs = torchaudio.load(audio_path)
    output_probs, score, index, text_lab = classifier.classify_batch(signal)
    csv_data.at[i, 'Label'] = float(text_lab[0])
csv_data.to_csv('datasheet.csv')
