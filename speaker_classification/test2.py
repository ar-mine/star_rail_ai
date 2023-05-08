from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import soundfile

inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)

speaker1_a_wav = soundfile.read('/mnt/e/Github/star_rail_ai/dataset/deciphered_wav/External0/External0 00006.wav', dtype="int16")[0]
speaker1_b_wav = soundfile.read('/mnt/e/Github/star_rail_ai/dataset/deciphered_wav/External0/External0 00013.wav', dtype="int16")[0]
speaker2_a_wav = soundfile.read('/mnt/e/Github/star_rail_ai/dataset/deciphered_wav/External0/External0 00039.wav', dtype="int16")[0]


# enroll = inference_sv_pipline(audio_in=speaker1_a_wav)["spk_embedding"]
#
# same = inference_sv_pipline(audio_in=speaker2_a_wav)["spk_embedding"]
#
# # 对相同的说话人计算余弦相似度
# sv_threshold = 0.9465
# same_cos = np.sum(enroll * same) / (np.linalg.norm(enroll) * np.linalg.norm(same))
# same_cos = max(same_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
# print(same_cos)

rec_result = inference_sv_pipline(audio_in=('/mnt/e/Github/star_rail_ai/dataset/deciphered_wav/External0/External0 00006.wav',
                                            '/mnt/e/Github/star_rail_ai/dataset/deciphered_wav/External0/External0 00039.wav'))
print(rec_result["scores"][0])
