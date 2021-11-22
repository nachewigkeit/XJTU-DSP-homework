import numpy as np
import wave


def wavDecode(dir: str) -> np.ndarray:
    """
    输入文件位置，返回解析后的序列
    注意返回为二维数组，第一维指定声道
    """
    try:
        f: wave.Wave_read = wave.open(dir, "rb")
    except BaseException:
        print("读取wav文件" + str(dir) + "错误")
        print("请检查路径是否正确")
        return

    cSz, binSz, rate, n, _, _ = f.getparams()
    # 声道数、位深、采样率、采样点数
    info = (cSz, binSz, rate, n)
    tmp = f.readframes(n)

    data: np.ndarray = np.frombuffer(tmp, dtype=np.short)
    data = data.reshape((n, cSz))
    return (info, data)


def frame(wave_data, wlen, inc):
    # 帧数
    signal_length = len(wave_data)
    if signal_length <= wlen:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))

    # 补齐后分帧
    pad_length = int((nf - 1) * inc + wlen)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((wave_data, zeros))
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    return frames


def window(frames, method="hanning"):
    lframe = frames.shape[1]
    if method == "hanning":
        windown = np.hanning(lframe)
        return frames * windown
