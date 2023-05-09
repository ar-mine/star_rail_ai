from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell",
                                           savedir="pretrained_models/asr-transformer-aishell_model")
speech_file = "/home/armine/Dataset/deciphered_wav/External0/External0 00001.wav"
result = asr_model.transcribe_file(speech_file)

print(result)
