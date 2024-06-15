import os
from lib.ham_ho_tro import save_obj, load_obj
from models import lstm, cnn
from realtime import Realtime


cwd = os.getcwd()

if 'intro_AI' not in cwd:
    raise Exception('open sai thư mục rồi má ôi. open thư mục intro_AI đi')



# =================================================================

# audio_features = ['zcr', 'chroma_stft', 'mfcc', 'rms', 'mel']
# a = cnn('TESS', '1d', audio_features)

# a.run(50)
# save_obj(a)

# # =================================================================


# b = lstm('TESS', 'time', 'mfcc', False)

# b.run(50)

# save_obj(b)

# =================================================================

# c = Realtime()
# c.start()




