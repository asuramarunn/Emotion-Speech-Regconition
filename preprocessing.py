import os, librosa, random
import numpy as np
from sklearn.preprocessing import StandardScaler
import noisereduce as nr



def handle_path(pl,  predict):
    '''hàm này có nhiệm vụ gán nhãn cho từng file âm thanh và cho vào một list
    '''
    dr = []
    for path in pl:
        cwd = os.getcwd()
        if 'intro_AI' not in cwd:
            print('Có lẽ bạn đang mở sai thư mục. Bạn hãy mở lại thư mục intro_AI cho đúng')
        path_data = os.path.join(cwd, 'data')
        if not os.path.isdir(path_data):
            raise NotADirectoryError('Thư mục data không tồn tại')
        if type(path) == str:
            path_list = [os.path.join(path_data, path)]
        elif type(path) == list or type(path) ==tuple:
            path_list = [os.path.join(path_data, i) for i in path]
        else:
            raise TypeError('path phải là str hoặc list hoặc tuple')
        if predict:
            for k in os.listdir(pl[0]):
                dr.append([os.path.join(pl[0], k), -1])
        else:
            for i in path_list:
                for j in range(7):
                    pl = os.path.join(i, decode_emotion(j))
                    try:
                        for k in os.listdir(pl):
                            dr.append([os.path.join(pl, k), j])
                    except:
                        pass
    random.shuffle(dr)
    return dr

def split_pad_data(data, y, n_sample):
    n_sample = int(n_sample)
    l = len(data)
    n = l//n_sample
    d = l%n_sample
    if d > 0.25*n_sample:
        n+=1
    if n==0:
        n=1
    dr = []
    for i in range(n):
        dt = data[i*n_sample:(i+1)*n_sample]
        padded_x = np.pad(dt, (0, n_sample-len(dt)), 'constant')
        dr.append([padded_x, y])
    return dr

def audio_preprocessing(path_data, select, n_sample, predict = False, datax = None):
    '''trả về một danh sách âm thanh đã được tiền sử lí'''
    sr = 22050
    if datax is None:
        x = handle_path(path_data, predict)
    else:
        # datax là một mảng chứa các phần tử là data của từng đoạn âm thanh
        x = [[i, -1] for i in datax]
    for i in x:
        if datax is None:
            # i có dạng: ['path', y]
            data, _ = librosa.load(i[0])
        else:
            data = i[0]
        if select['sequentially']:
            if select['normalize']:
                data = librosa.util.normalize(data, norm=+5.0)
            if select['trim']:
                data = librosa.effects.trim(data, top_db=20)[0]
            if select['reduce_noise']:
                # An: Hàm này giảm các tạp âm đi
                # An: T test thử với phổ mfcc thì thấy các tạp âm được giảm khá rõ, nhưng đồng thời
                # An: ở các vùng có âm thanh con người sẽ bị giảm giá trị tính được 1 chút, không quá đáng kể
                data = nr.reduce_noise(data,sr=22050)
            if select['stretch']:
                max_duration = 2
                duration = len(data) / sr
                # An: Kiểm tra độ dài của đoạn âm
                if duration < max_duration:
                # An: Áp dụng time stretching nếu đoạn âm ngắn hơn 2 giây
                    data = librosa.effects.time_stretch(data, rate=0.6)
            if select['picth']:
                # An: tăng hoặc giảm cường độ đầu vào, n_steps < 1 giảm, > 1 tăng
                data = librosa.effects.pitch_shift(data, sr = sr, n_steps = 0.8)
            if select['add_noise']:
                # An: thêm nhiễu
                noise = 0.035*np.random.uniform()*np.amax(data)
                data = data + noise*np.random.normal(size=data.shape[0])
            if select['shift']:
                # An: dịch 1 khoảng  
                shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                data = np.roll(data, shift_range)
            if select['remove_slient']:
                # An: hàm này rất dễ cắt mất phần lớn các đoạn âm nếu dùng không cẩn thận, nên lựa chọn các âm bé
                # An: lọc những đoạn âm nhỏ hơn 10db trong dữ liệu
                data_slient = librosa.effects.split(data,top_db=10)
                # An: lấy những đoạn âm còn lại
                non_silent_data = [data[start:end] for start, end in data_slient]
                # An: ghép lại thành dữ liệu hoàn chỉnh
                data = np.concatenate(non_silent_data)
            if select['split_pad_data']:
                xy = split_pad_data(data, i[1], n_sample)
                for j in xy:
                    yield j
            else:
                yield [data, i[1]]
        else:
            yield [data, i[1]]
            if select['normalize']:
                normalize_data = librosa.util.normalize(data, norm=+5.0)
                if select['split_pad_data']:
                    xy = split_pad_data(normalize_data, i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield [normalize_data,i[1]]
            if select['trim']:
                trim_data = librosa.effects.trim(data, top_db=20)[0]
                if select['split_pad_data']:
                    xy = split_pad_data(trim_data, i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield [trim_data,i[1]]
            if select['add_noise']:
                noise = 0.035*np.random.uniform()*np.amax(data)
                noise = data + noise*np.random.normal(size=data.shape[0])
                if select['split_pad_data']:
                    xy = split_pad_data(noise, i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield (noise,  i[1])
            if select['stretch']:
                max_duration = 2
                duration = len(data) / sr
                    # An: Kiểm tra độ dài của đoạn âm
                if duration < max_duration:
                    # An: Áp dụng time stretching nếu đoạn âm ngắn hơn 2 giây
                    stretch_data = librosa.effects.time_stretch(data, rate=0.6)
                else:
                    # An: Trả lại đoạn âm gốc nếu độ dài lớn hơn hoặc bằng 2 giây
                    stretch_data = data
                if select['split_pad_data']:
                    xy = split_pad_data(stretch_data, i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield [stretch_data, i[1]]
            if select['picth']:
                picth = librosa.effects.pitch_shift(data, sr = 22050, n_steps = 1)
                if select['split_pad_data']:
                    xy = split_pad_data(picth, i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield (picth,  i[1])
                if select['stretch']:
                    stretch_data = librosa.effects.time_stretch(data, rate=0.6)
                    x = librosa.effects.pitch_shift(stretch_data, sr = 22050, n_steps = 1)
                    if select['split_pad_data']:
                        xy = split_pad_data(x, i[1], n_sample)
                        for j in xy:
                            yield j
                    else:
                        yield (x,  i[1])
            if select['shift']:
                shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                if select['split_pad_data']:
                    xy = split_pad_data(np.roll(data, shift_range), i[1], n_sample)
                    for j in xy:
                        yield j
                else:
                    yield (np.roll(data, shift_range),  i[1])


def one_d(audio_features, ds):
    scaler = StandardScaler()
    X = []
    for dc in ds:
        t = [dc[i] for i in audio_features]
        re = np.vstack(t)
        tb = np.mean(re, axis=1)
        X.append(tb)
    X = scaler.fit_transform(X)
    X = np.expand_dims(X, axis=2)
    return X

def two_d(audio_features, X):
    return l_time(audio_features, X, False)

def l_freq(audio_features, ds, TB):
    X = []
    for dc in ds:
        t = [dc[i] for i in audio_features]
        re = np.vstack(t)
        if TB:
            x = np.expand_dims(np.mean(re, axis=1), -1)
            X.append(x)
            # mai thêm cái cho các phân từ trong 
        else:
            X.append(re)
    X = np.asarray(X)
    return X

def l_time(audio_features, ds, TB):
    X = []
    for dc in ds:
        if type(TB) == bool:
            if TB:
                if 'chroma_stft' in audio_features:
                    dc['chroma_stft'] = np.mean(dc['chroma_stft'], axis=0)
                if 'mfcc' in audio_features:
                    dc['mfcc'] = np.mean(dc['mfcc'], axis=0)
                if 'mel' in audio_features:
                    dc['mel'] = np.mean(dc['mel'], axis=0)
            t = [dc[i] for i in audio_features]
            re = np.vstack(t)
            X.append(re.T)
        elif type(TB) == tuple or type(TB) == list:
            if TB[0] and 'chroma_stft' in audio_features:
                dc['chroma_stft'] = np.mean(dc['chroma_stft'], axis=0)
            if TB[1] and 'mfcc' in audio_features:
                dc['mfcc'] = np.mean(dc['mfcc'], axis=0)
            if TB[2] and 'mel' in audio_features:
                dc['mel'] = np.mean(dc['mel'], axis=0)
            t = [dc[i] for i in audio_features]
            re = np.vstack(t)
            X.append(re.T)
        else:
            raise ValueError("với time_or_frequency == 'time' TB chỉ nhận đối số đầu vào là bool, tuple hoặc list")
    X = np.asarray(X)
    return X

def extract_features(data, audio_features, n_chroma, n_mfcc, hs):
    dc = {}
    sr = 22050
    if 'zcr' in audio_features:
        zcr = librosa.feature.zero_crossing_rate(y=data)
        dc['zcr'] = zcr*hs[0]
    if 'chroma_stft' in audio_features:
        stft = librosa.stft(data)
        chroma_stft = np.abs(librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=n_chroma))
        dc['chroma_stft'] = chroma_stft*hs[1]
    if 'mfcc' in audio_features:
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
        dc['mfcc'] = mfcc*hs[2]
    if 'rms' in audio_features:
        rms = librosa.feature.rms(y=data)
        dc['rms'] = rms*hs[3]
    if 'mel' in audio_features:
        mel = librosa.feature.melspectrogram(y=data, sr=sr)
        dc['mel'] = mel*hs[4]
    return dc

def audio_feature(sound_list, cf, audio_features, n_chroma, n_mfcc, hs, TB=False):
    '''Hàm này giúp trích xuất các đặc trưng của âm thanh
    Lưu ý: ma trận đầu ra có dạng (dc, t). với dc là các đặc trưng, t là thời gian
    nếu muốn chuyển vị ma trận thì thực hiện với kết quả của hàm này'''
    X = []
    y = []
    
    for i in sound_list:
        X.append(extract_features(i[0], audio_features, n_chroma, n_mfcc, hs))
        y.append(encode_number(i[1]))
    
    Y = np.asarray(y)
    if cf == '1d':
        return one_d(audio_features, X), Y
    elif cf == '2d':
        return two_d(audio_features, X), Y
    elif cf == 'frequency':
        return l_freq(audio_features, X, TB), Y
    elif cf == 'time':
        return l_time(audio_features, X, TB), Y
    else:
        raise ValueError('cf không hợp lệ')
    

def encode_emotion(emotional):
    match emotional:
        case 'neutral':
            return 0
        case 'happy':
            return 1
        case 'sad':
            return 2
        case 'angry':
            return 3
        case 'fear':
            return 4
        case 'disgust':
            return 5
        case 'surprise':
            return 6
        case _:
            raise ValueError('emotional không hợp lệ')

def decode_emotion(encode_emotion):
    match encode_emotion:
        case 0:
            return 'neutral'
        case 1:
            return 'happy'
        case 2:
            return 'sad'
        case 3:
            return 'angry'
        case 4:
            return 'fear'
        case 5:
            return 'disgust'
        case 6:
            return 'surprise'
        case _:
            return ''

def decode_emotion_vi(encode_emotion):
    match encode_emotion:
        case 0:
            return 'trung lập'
        case 1:
            return 'vui vẻ'
        case 2:
            return 'buồn'
        case 3:
            return 'tức giận'
        case 4:
            return 'sợ hãi'
        case 5:
            return 'cảm thấy ghê'
        case 6:
            return 'ngạc nhiên'
        case _:
            return ''

def encode_number(number):
    if number == -1:
        return -1
    encoded_vector = np.zeros(7)
    encoded_vector[number] = 1
    return encoded_vector

def decode(list_encoded_vector):
    dr = []
    for i in list_encoded_vector:
        dr.append(np.argmax(i))
    return dr

