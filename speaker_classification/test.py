# Reference: https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/quickstart
from modelscope.pipelines import pipeline

sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)
speaker1_a_wav = 'dataset/deciphered_wav/External0/External0 00006.wav'
speaker1_b_wav = 'dataset/deciphered_wav/External0/External0 00013.wav'
speaker2_a_wav = 'dataset/deciphered_wav/External0/External0 00019.wav'
# 相同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker1_b_wav])
print(result)
# 不同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav])
print(result)
# 可以自定义得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav], thr=0.31)
print(result)
