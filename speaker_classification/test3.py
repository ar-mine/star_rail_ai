from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")
speaker_a_1 = "/home/armine/Dataset/deciphered_wav/External0/External0 00006.wav"
speaker_a_2 = "/home/armine/Dataset/deciphered_wav/External0/External0 00025.wav"

score, prediction = verification.verify_files(speaker_a_1, speaker_a_2)


print(prediction, score)
