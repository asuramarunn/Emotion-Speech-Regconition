import os
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from lib.preprocessing import *
from lib.ham_ho_tro import *




class Model:
    '''class này để cho các class con kế thừa'''
   
    def __init__(self):
        pass
    
    def build_model(self):
        pass
    
    def preprocessing(self):
        pass
    def train(self):
        pass

    def resume_training(self, epochs_to_train):
        if self.continue_studying == False:
            raise Exception('Bạn đã cài đặt không học tiếp ngay từ đầu, và self.X, self.Y đã được xóa đi để tiếc kiệm bộ nhớ')
        # Đảm bảo có sự huấn luyện trước đó
        if self.history is None:
            print("Không có quá trình huấn luyện trước đó để tiếp tục.")
            return
        if epochs_to_train < self.history.epoch[-1]:
            print("Số epoch cần tiếp tục huấn luyện phải lớn hơn số epoch đã huấn luyện trước đó.")
            print("Số epoch đã huấn luyện trước đó:", self.history.epoch[-1])
            print("Số epoch cần tiếp tục huấn luyện:", epochs_to_train)
            print("hay là bạn chạy lại từ đầu 😃")
            return None
        try:
            # Tiếp tục huấn luyện từ epoch cuối cùng của lần trước
            self.history = self.model.fit(self.X, self.Y, validation_split=0.2, epochs = epochs_to_train, batch_size=64, initial_epoch=self.history.epoch[-1]+1)
            print("Tiếp tục huấn luyện thành công.")
        except Exception as e:
            print("Đã xảy ra lỗi khi tiếp tục huấn luyện:", str(e))
    
    def results(self):
        plot_acc_and_loss(self.history, self.epochs)
        ev = self.evaluate(data=(self.X_test, self.Y_test))
        print('\n', ev[0], '\n')
        print(ev[1], '\n')
        print(ev[2], '\n')
        plt.figure(figsize=(12,8))
        ax = plt.axes()
        sns.heatmap(ev[1], ax = ax, cmap = 'BuGn', fmt="d", annot=True)
        ax.set_ylabel('True emotion')
        ax.set_xlabel('Predicted emotion')
        plt.show()
        if self.continue_studying == False:
            del self.X_test, self.Y_test  # giải phóng bộ nhớ
    
    def predict(self):
        pass

    def evaluate(self, path=None, data=None):
        '''thư mục cần kiểm tra đặt trong thư mục data, chỉ cần ghi mình tên thư mục, không cần ghi đường dẫn đầy đủ'''
        if type(path) == str:
            path = [path]
        if path != None:
            sound_list = audio_preprocessing(path, self.select_audio_preprocessing, self.n_sample)
            X, Y = audio_feature(sound_list, self.time_or_frequency, self.audio_features, self.n_chroma, self.n_mfcc, self.hs, self.TB)
            y_pred = self.predict(self, data=X)
        elif data != None:
            X, Y = data
            y_pred = decode(self.model.predict(X))
        else:
            raise Exception('Bạn chưa chọn đường dẫn hoặc dữ liệu để dự đoán')
        eva = self.model.evaluate(X, Y, verbose=2)
        y = decode(Y)
        mlb = list(set(y))
        mlb.sort()
        lb=[]
        for i in mlb:
            lb.append(decode_emotion(i))
        cm=confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, lb, lb)                      
        return eva, cm_df, classification_report(y, y_pred, target_names=lb)

    def run(self, epochs = 50):
        self.preprocessing()
        self.train(epochs)
        self.results()



class cnn(Model):
    '''
    path_data: là đường dẫn tương đối tới dữ liệu đầu vào, có thể là đến một 
    hay nhiều thư mục nếu nhiều thư mục thì đặt trong một list
    
    num_dimensions: quyết định xem cách học mô hình cnn là học dạng ảnh (2d) hay lấy trung bình
    các tính chất (1d), nó nhận 2 giá trị là num_dimensions = '1d' hoặc num_dimensions = '2d'
    
    audio_features: là các đặc trưng âm thanh bạn muốn đưa vào để học. vd: audio_features = ('mfcc', 'zcr', 'mel')
    
    '''
    continue_studying = False
    n_chroma = 12
    n_mfcc = 20
    n_sample = 5*22050
    hs = [1, 1, 1, 1, 1]
    
    select_audio_preprocessing = {
        'sequentially': True, # default
        'normalize': True,
        'trim': True,
        'reduce_noise': False,
        'stretch': False,
        'picth' : False,
        'add_noise': False,
        'shift': False,
        'remove_slient' : False, # đọc lại cảnh báo của An
        'split_pad_data': True
    }
    
    
    def __init__(self, path_data, num_dimensions = None, audio_features = None):
        self.path_data = path_data
        self.num_dimensions = num_dimensions
        self.audio_features = audio_features
        if type(audio_features) == str:
            self.audio_features =[audio_features]
        if type(path_data) == str:  
            if os.path.isfile(path_data):
                raise ValueError('Đường dẫn phải là một thư mục')
            self.path_data = [path_data]
        input_shape = cal_input_shape(num_dimensions, audio_features, self.n_sample, self.n_chroma, self.n_mfcc)
        print('\nkích thước ma trận đầu vào đã tính toán:', input_shape, '\n\n')
        self.loss = 'categorical_crossentropy'
        self.build_model(input_shape)
    
    def build_model(self, input_shape):
        if self.num_dimensions == '1d':
            model=Sequential()
            model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

            model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

            model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
            model.add(Dropout(0.2))

            model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

            model.add(Flatten())
            model.add(Dense(units=32, activation='relu', kernel_regularizer = regularizers.l1(0.01)))
            model.add(Dropout(0.3))

            model.add(Dense(units=7, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
            model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

            self.model = model
            
        elif self.num_dimensions == '2d':
            # ...
            pass
        else:
            raise ValueError('num_dimensions chỉ nhận 2 giá trị là 1d hoặc 2d')
    
    def preprocessing(self):
        sound_list = audio_preprocessing(self.path_data, self.select_audio_preprocessing, self.n_sample)
        X, Y = audio_feature(sound_list, self.num_dimensions, self.audio_features, self.n_chroma, self.n_mfcc, self.hs)
        print('\nkích thước ma trận qua bộ tiền sử lí:', X.shape, Y.shape)
        print('Nếu khác kích thước ma trận đã tính toán thì có vấn đề\n')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = 0.2)

    def train(self, epochs = 100):
        self.epochs = epochs
        rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=64, epochs=epochs, validation_data=(self.X_test, self.Y_test), callbacks=[rlrp])
        # self.model.save('storage/LSTM/LSTM.keras') # không dùng cách này để lưu mô hình nữa
        if self.continue_studying == False:
            del self.X_train, self.Y_train  # giải phóng bộ nhớ

    def predict(self, path=None, data=None):
        '''file nhớ ghi đầy đủ đường dẫn, có cả chữ data nếu đặt trong thư mục data'''
        if path is None and data is None:
            raise Exception('Bạn chưa chọn đường dẫn hoặc dữ liệu để dự đoán')
        if type(path) == str:
            path = [path]
        sound_list = audio_preprocessing(path, self.select_audio_preprocessing, self.n_sample, predict = True, datax = data)
        X, _ = audio_feature(sound_list, self.num_dimensions, self.audio_features, self.n_chroma, self.n_mfcc, self.hs)
        return decode(self.model.predict(X))



class lstm(Model):
    '''
    path_data: là đường dẫn tương đối tới dữ liệu đầu vào, có thể là đến một 
    hay nhiều thư mục nếu nhiều thư mục thì đặt trong một list.
    
    time_or_frequency: bạn có thể chọn học theo trục thời gian hay tần số, nó nhận 2 giá trị
    là time_or_frequency = 'time' hoặc time_or_frequency = 'frequency'
    
    audio_features: là các đặc trưng âm thanh bạn muốn đưa vào để học. vd: audio_features = ('mfcc', 'zcr', 'mel')
    
    TB: khi time_or_frequency = 'frequency' TB mang 2 giá trị TB = True hoặc TB = False
    khi time_or_frequency = 'time' thì TB có thể mang giá trị True hoặc False hoặc tuple hay list
    các giá trị bool đại diện cho tính trung bình của (chroma_stft, mfcc, mel). vd: TB = (True, False, True)
    '''
    continue_studying = False
    n_chroma = 12
    n_mfcc = 20
    n_sample = 5*22050
    hs = [1, 1, 1, 1, 1]   # hệ số nhân với các tính chất: zcr, chroma_stft, mfcc, rms, mel
    
    select_audio_preprocessing = {
        'sequentially': True, # default
        'normalize': True,
        'trim': True,
        'reduce_noise': False,
        'stretch': False,
        'picth' : False,
        'add_noise': False,
        'shift': False,
        'remove_slient' : False, # đọc lại cảnh báo của an
        'split_pad_data': True
    }
    
    
    def __init__(self, path_data, time_or_frequency = None, audio_features = None, TB = False):
        self.path_data = path_data
        self.time_or_frequency = time_or_frequency
        self.audio_features = audio_features
        self.TB = TB
        if type(audio_features) == str:
            self.audio_features =[audio_features]
        if type(path_data) == str:  
            if os.path.isfile(path_data):
                raise ValueError('Đường dẫn phải là một thư mục')
            self.path_data = [path_data]
        input_shape = cal_input_shape(time_or_frequency, audio_features, self.n_sample, self.n_chroma, self.n_mfcc, TB)
        print('\nkích thước ma trận đầu vào đã tính toán:', input_shape, '\n\n')
        self.loss = 'categorical_crossentropy'
        self.build_model(input_shape)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(256, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(7, activation='softmax')
        ])
        model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])
        self.model = model
    
    def preprocessing(self):
        sound_list = audio_preprocessing(self.path_data, self.select_audio_preprocessing, self.n_sample)
        X, Y = audio_feature(sound_list, self.time_or_frequency, self.audio_features, self.n_chroma, self.n_mfcc, self.hs, self.TB)
        print('\nkích thước ma trận qua bộ tiền sử lí:', X.shape, Y.shape)
        print('Nếu khác kích thước ma trận đã tính toán thì có vấn đề\n')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = 0.2)

    def train(self, epochs = 100):
        self.epochs = epochs
        self.history = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_test, self.Y_test), epochs=epochs, batch_size=64)
        # self.model.save('storage/LSTM/LSTM.keras') # không dùng cách này để lưu mô hình nữa
        if self.continue_studying == False:
            del self.X_train, self.Y_train  # giải phóng bộ nhớ
    
    def predict(self, path=None, data=None):
        '''file nhớ ghi đầy đủ đường dẫn, có cả chữ data nếu đặt trong thư mục data'''
        if path is None and data is None:
            raise Exception('Bạn chưa chọn đường dẫn hoặc dữ liệu để dự đoán')
        if type(path) == str:
            path = [path]
        sound_list = audio_preprocessing(path, self.select_audio_preprocessing, self.n_sample, predict = True, datax = data)
        X, _ = audio_feature(sound_list, self.time_or_frequency, self.audio_features, self.n_chroma, self.n_mfcc, self.hs, self.TB)
        return decode(self.model.predict(X))




