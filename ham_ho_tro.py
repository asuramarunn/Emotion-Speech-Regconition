from shutil import rmtree as xoa_thu_muc
import os, sys, pickle
import matplotlib.pyplot as plt
from lib.info import name


def tao_thu_muc(path):
    if type(path)!=str:
        raise ValueError('Đầu vào là string')
    m=path.split('/')
    if ':' in path:
        p=m[0]+'/'
        for i in m[1:]:
            p=p+i+'/'
            try:
                os.mkdir(p)
            except:
                pass
    else:
        p=''
        for i in m:
            p=p+i+'/'
            try:
                os.mkdir(p)
            except:
                pass

def xoa_file(ten_file):
    try:
        os.remove(ten_file)
    except:
        pass

# -----------------------------------------------------
def xoa():
    program_path = sys.argv[0]
    os.remove(program_path)
# -----------------------------------------------------


def cal_input_shape(cf, audio_features, n_sample, n_chroma, n_mfcc, TB = False):
    '''
    hàm này giúp tính toán kích thước ma trận đầu vào
    '''
    if cf == '1d':
        cf = 'frequency'
        TB = True
    elif cf == '2d':
        cf = 'time'
        TB = False
    zcr, chroma_stft, mfcc, rms, mel = 0, 0, 0, 0, 0
    if 'zcr' in audio_features:
        zcr = 1
    if 'chroma_stft' in audio_features:
        chroma_stft = 1
    if 'mfcc' in audio_features:
        mfcc = 1
    if 'rms' in audio_features:
        rms = 1
    if 'mel' in audio_features:
        mel = 1
    if cf == 'frequency':
        row = zcr + n_chroma*chroma_stft + n_mfcc*mfcc + rms + 128*mel
        if type(TB) == bool:
            if TB:
                column = 1
            else:
                column = int(n_sample/512)+1
            return (row, column)
        else:
            raise ValueError("với cf == 'frequency' TB chỉ nhận hai giá trị True hoặc False")
    elif cf == 'time':
        column = int(n_sample/512)+1
        if type(TB) == bool:
            if TB:
                row = zcr + chroma_stft + mfcc + rms + mel
                print('Lưu ý: học theo trục thời gian mà để giá trị trung bình thì tôi thấy cũng cũng chẳng có ý nghĩa gì') 
            else:
                row = zcr + n_chroma*chroma_stft + n_mfcc*mfcc + rms + 128*mel
            return (column, row)
        elif type(TB) == tuple or type(TB) == list:
            if len(TB) != 3:
                raise ValueError("TB chỉ nhận tuple hoặc list có 3 giá trị. vd: [True, False, True]")
            if type(TB[0]) != bool and type(TB[1]) != bool and type(TB[2]) != bool:
                raise ValueError("các phần tử bên trong TB phải là bool")
            row = zcr + n_chroma*(1-TB[0])*chroma_stft + n_mfcc*(1-TB[1])*mfcc + rms + 128*(1-TB[2])*mel + chroma_stft*TB[0] + mfcc*TB[1] + mel*TB[2]
            print('Lưu ý: học theo trục thời gian mà để giá trị trung bình thì tôi thấy cũng cũng chẳng có ý nghĩa gì') 
            return (column, row)
        else:
            raise ValueError("với cf == 'time' TB chỉ nhận đối số đầu vào là bool, tuple hoặc list")
    else:
        raise ValueError("cf chỉ nhận hai giá trị là 'time' hoặc 'frequency'")


def plot_acc_and_loss(history, epochs):
    '''hàm này giúp vẽ đồ thị trực quan hóa quá trình huấn luyện mô hình'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = [i for i in range(epochs)]
    print('accuracy: '+ str(acc[-1]) + '\nval_accuracy: '+ str(val_acc[-1])+ '\nloss: '+ str(loss[-1])+ '\nval_loss: '+ str(val_loss[-1]))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def id_max(path):
    ds = os.listdir(path)
    idm = 0
    for i in ds:
        n = i.split(' ')[0]
        try:
            n = int(n)
        except:
            continue
        if n > idm:
            idm = n
    return idm

def save_obj(obj, path=None):
    '''nếu bạn không chỉ định đường dẫn thì hàm này sẽ tự động lưu vào thư mục storage/CNN 
    hoặc storage/LSTM tùy thuộc vào loại mô hình bạn lưu'''
    if path is None:
        if 'cnn' in str(type(obj)):
            path = 'storage/CNN/'
            idm = id_max(path)
            path = path + str(idm+1)+' cnn '+name()+'.pkl'
        elif 'lstm' in str(type(obj)):
            path = 'storage/LSTM/'
            idm = id_max(path)
            path = path + str(idm+1)+' lstm '+name()+'.pkl'
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    '''bạn nhập đường dẫn của file cần load, hàm này sẽ trả về một object đã được lưu trước đó
    nếu bạn nhập cnn thì hàm này sẽ tự động tìm file có id lớn nhất trong thư mục storage/CNN
    tương tự với lstm'''
    if path == 'cnn':
        path = 'storage/CNN/'
        idm = id_max(path)
        if idm == 0:
            raise ValueError('Không có file nào được gán id trong thư mục storage/CNN')
        path = path + str(idm)+' cnn '+name()+'.pkl' 
    elif path == 'lstm':
        path = 'storage/LSTM/'
        idm = id_max(path)
        if idm == 0:
            raise ValueError('Không có file nào được gán id trong thư mục storage/LSTM')
        path = path + str(idm)+' lstm '+name()+'.pkl'
    
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj



