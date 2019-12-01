    # -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:07:54 2019

@author: 247
"""

  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:15:13 2018
@author: fadlimuharram
"""

# import the necessary packages
from time import time
import tensorflow
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import flask
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from flask_cors import CORS
import numpy
from sklearn import metrics


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)
klasifikasi = None
train_set = None
test_set = None
datanya = None
jumlahKelas = None

'''development atau production atau initial'''

MODENYA = None 
productionEpochnya = None

print('')
print('Input Dengan Menggunakan Model Yang Telah Tersedia')
print('[1] Tidak')
print('[2] Ya')
isLoadedDariModel = int(input('Input Pilihan : '))
print(isLoadedDariModel, type(isLoadedDariModel))
print(isLoadedDariModel != 1)
if isLoadedDariModel != 2 and isLoadedDariModel != 1 :
    raise ValueError('Error ! Mohon Input Angka 1 dan 2')
if isLoadedDariModel == 2:
    isLoadedDariModel = True
    print('Masukan Jumlah Epoch Sebelumnya')
    productionEpochnya = int(input('Input Jumlah Epoch : '))
elif isLoadedDariModel == 1:
    isLoadedDariModel = False
    print(' ')
    print('Pilih Mode Training')
    print('[1] Production')
    MODENYA = int(input('Input Pilihan : '))
    if MODENYA == 1:
        MODENYA = 'production'
    else:
        raise ValueError('Error ! Mohon Input Angka 1')
    
    if MODENYA == 'production':
        print('Pilih Jumlah Epoch Yang Di Ingin Dijalankan')
        productionEpochnya = int(input('jumlah Epoch : '))
        
else:
    raise ValueError('Error ! Mohon Input 0 dan 1')


#isLoadedDariModel = True
#productionEpochnya = 5

IPNYA = '127.0.0.1'
PORTNYA = 5000


LOKASI_TRAINING = 'D:/Tugas Unas/TA/PlantVillage/Latihan/training'
LOKASI_TESTING = 'D:/Tugas Unas/TA/PlantVillage/Latihan/testing'


LOKASI_UPLOAD = 'upload'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','JPG'])

PENYAKIT_TANAMAN = {
        0: {
                "nama":'''<h2 style="text-align: center;">Apple Scab</h2>
                ''',
                "penyebab":'''<h4>Penyebab</h4>'''
                '''<p>Jamur <i>Venturia inaequalis</i></p>''',
                "gejala":'''<h4>Gejala</h4>'''
                '''
                <p style="text-align: justify;">Gejala penyakit yang paling  umum  dijumpai dan dianggap penting adalah pada daun dan buah. Bercak  daun dapat terjadi pada kedua sisi permukaan daun. Namun hal  ini masih tergantung pada waktu terjadinya infeksi. Waktu  tunas membuka  maka permukaan bawah daun yang lebih dahulu  terbuka,  disamping  itu permukaan tersebut lebih  basah  sehingga  lebih mudah terinfeksi. 
                Seiring berkembangnya penyakit warna bercak ini akan berubah menjadi hitam keunguan</p>
                ''',
                "penangan":'''<h4>Penanganan</h4>'''
                '''
                <ul style="text-align: justify;">
                    <li>Pilih varietas yang lebih kuat bila memungkinkan.</li>
                    <li>Beri air di malam atau pagi hari (menghindari biaya overhead irigasi) untuk memberikan waktu daun mengering sebelum infeksi dapat terjadi.</li>
                    <li>Penyemprotan dengan fungisida  seperti  bubur Bordeaux  masih  cukup efektif; fungisida  lain  yang  dapat dianjurkan adalah captan, ferbam, glyodin dan jenis belerang.</li>
                    <li>Untuk kontrol terbaik, semprot sabun tembaga cair awal, dua minggu sebelum gejala biasanya muncul. Atau, mulai aplikasi ketika penyakit muncul pertama, dan ulangi pada 7 sampai 10 interval hari sampai dengan penurunan bunga.</li>
                </ul>
                '''
            },
        1: {
                "nama":'''<h2 style="text-align: center;">Black Rot</h2>''',
                "penyebab":'''<h4>Penyebab</h4>'''
                '''<p>jamur <i>Botryosphaeria obtusa</i></p>''',
                "gejala":'''<h4>Gejala</h4>'''
                '''
                <p style="text-align: justify;">Gejala awal terlihat umumnya pada buah yang menjelang masak berupa luka berupa bercak coklat, kecil dan kemudian meluas hingga warnanya berubah semakin gelap atau hitam. Pada buah yang berwarna hijau bisanya bercaknya langsung hitam. Pada bagian pusat bercak sering nampak garis-garis konsentris. Buah yang sakit busuk tidak berbau, dan permukaan yang busuk tetap rata. Bila busuk telah sempurna pada buah biasanya akan mengering, jaringan melipat, kadang-kadang menjadi mengeras (mumifikasi). Pada saat demikian biasanya patogen membentuk badan buah yang berupa bintik-bintik hitam.
            Gejala pada daun diawali dengan adanya bintik kecilberwarna ungu, kemudian segera meluas, dengan garis tengah bercak sekitar 2-10 mm atau rata-rata 4 mm. Bercak berbentuk bulat mempunyai garis batas yang jelas, lama kelamaan bagian tengahnya menjadi berwarna coklat kekuningan. Apabila terjadi serangan sekunder bentuk bercak menjadi tidak teratur. Pada pusat bercak dapat dilihat adanya bintik-bintik hitam yang merupakan kumpulan piknidium patogen.</p>
                ''',
                "penangan":'''
                <p style="text-align: justify;">Mengobati black rot pada pohon apel dimulai dengan sanitasi. Karena spora jamur menahan musim dingin pada daun jatuh, kulit mati dan Kanker, penting untuk menjaga semua puing-puing jatuh dan buah mati dibersihkan jauh dari pohon. Selama musim dingin, memeriksa merah membusuk dan menghapus mereka dengan memotong mereka keluar atau pemangkasan jauh tungkai yang terkena setidaknya enam inci di luar luka. Menghancurkan semua jaringan yang terinfeksi segera dan menjaga pengawasan yang ketat untuk tanda-tanda baru infeksi. Setelah penyakit black rot di bawah kontrol di pohon Anda dan Anda lagi panen buah-buahan yang sehat, pastikan untuk membuang setiap buah terluka atau bekas gigitan serangga untuk menghindari infeksi ulang. Meskipun tujuan umum fungisida, seperti semprotan berbasis tembaga dan belerang kapur, dapat digunakan untuk mengendalikan black rot, tidak akan meningkatkan apel black rot seperti menghapus semua sumber spora.</p>
                '''
            },
        2: {
                "nama":'''<h2 style="text-align: center;">Cedar Apple Rust</h2>''',
                "penyebab":'''<h4>Penyebab</h4>'''
                '''<p>Jamur <i>Gymnosporangium juniperi-virginianae</i></p>''',
                
                "gejala":'''<h4>Gejala</h4>'''
                '''
                <p style="text-align: justify;">Gejala yang paling mencolok pada apel adalah berupa warna oranye terang, lukanya pada daun berkilau. Lesio yang tidak dihambat bahan kimia dapat membentuk berkas-berkas kecil yang menghasilkan struktur spora (aecia) pada permukaan bawah daun pada bulan Juli atau Agustus. 
                Karat cedar-apel muncul pada buah pertama sebagai lesiooranye terang, sedikit terangkat, tetapi dengan membesarnya buah penampilannya akan lebih coklat dan retak. Biasanya beberapa warna oranye tetap dipanen sebagai bukti infeksi awal musim. Sporulasi lesi pada buah kurang umum daripada lesi daun. 
                Batang terinfeksi menyebabkan pembengkakan batang dan dapat mengakibatkan amputasi buah muda. 
                Pada pohon cedar, karat cedar-apel memproduksi gall bulat coklat, ukuran diameternyamulai dari 6-7 mm sampai hampir 50 mm.</p>
                ''',
                "penangan":'''<h4>Penanganan</h4>'''
                '''
                <p style="text-align: justify;">Lakukan monitoring dengan memperhatikan keberadaan pohon-pohon cedar merah dalam jarak 1km dari kebun apel dan adanya karat Quince dan lakukan survei keberadaan gall karat cedar. Amati lesion yang mulai muncul 10 sampai 14 hari setelah terinfeksi.
                Kumpulkan gall karat dari pohon cedar merah dan uji apakah masih mampu bersporulasi dengan menempatkannya dalam air dalam cangkir putih. 
                Jika gall dengan lamanya pembasahan air berwarna oranye dalam beberapa jam, maka galls masih mampu menghasilkan spora selama periode pembasahan mendatang. 
                Bercak karat muncul pada daun dan buah masing-masing dalam waktu sekitar 14 hari dan dua sampai empat minggu setelah infeksi. 
                Fungisida yang efektif terhadap penyakit karat harus diterapkan secara periodik dari tahap perkembangan kuncup berwarna pink untuk melindungi muncul dan berkembangnya buah dan daun. 
                Menghilangkan sumber inokulum yang terletak dalam radius 2 mil dari kebun akan memotong siklus hidup jamur dan membuat kontrol dengan fungisida lebih mudah. 
                Menghilangkan semua pohon aras dalam waktu 4 sampai 5 mil dalam areal akan memberikan kontrol penuh.</p>
                '''
            },
        3: {
                "nama":'''<h2 style="text-align: center;">Healthy</h2>''',
                "gejala":'''
                <p style="text-align: justify;">Tetap jaga kesehatan pada pohon. Beri pupuk dan air dengan  teratur</p>
                ''',
                "penangan":'''
                
                '''
            }
        }

print(PENYAKIT_TANAMAN[0])
def hitungGambar(path):
    count = 0
    for filename in os.listdir(path):
        if filename != '.DS_Store':
            count = count + len(os.listdir(path+'/'+filename))
    return count

def hitungKelas():
    global LOKASI_TRAINING, LOKASI_TESTING, PENYAKIT_TANAMAN
    kelasTraining = 0
    kelasTesting = 0
    
    for filename in os.listdir(LOKASI_TRAINING):
        if filename != '.DS_Store':
            kelasTraining = kelasTraining + 1
            
    for filename in os.listdir(LOKASI_TESTING):
        if filename != '.DS_Store':
            kelasTesting = kelasTesting + 1
            
    if kelasTesting == kelasTraining and kelasTraining == len(PENYAKIT_TANAMAN) and kelasTesting == len(PENYAKIT_TANAMAN):
        return kelasTraining
    else:
        raise ValueError('Error: Kelas Training tidak sama dengan Kelas Testing')
        


app.config['UPLOAD_FOLDER'] = LOKASI_UPLOAD
app.config['STATIC_FOLDER'] = LOKASI_UPLOAD
jumlahKelas = hitungKelas()

    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def load_model_klasifikasi():
    global klasifikasi, train_set, test_set, datanya, kelasnya, LOKASI_TRAINING, LOKASI_TESTING
    global MODENYA, productionEpochnya, isLoadedDariModel
    # Initializing the CNN
    klasifikasi = Sequential()
    
    #input the first convolution with 32 filter, (5x5) kernel, and input shape (64,64,3)
    klasifikasi.add(Convolution2D(32,    # number of filter layers
                        5,    # y dimension of kernel (we're going for a 3x3 kernel)
                        5,    # x dimension of kernel
                        input_shape=(64, 64, 3),
                        init='he_normal'))
    
    # input activation function relu
    klasifikasi.add(Activation('relu'))
    
    # input subsampling layer (maxpooling) (2x2) for reduce data
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    
    #input the second convolution with 32 filter, (5x5) kernel, 
    klasifikasi.add(Convolution2D(64,    # number of filter layers
                            5,    # y dimension of kernel (we're going for a 3x3 kernel)
                            5,    # x dimension of kernel
                            init='he_normal'))
    
      # input activation function relu
    klasifikasi.add(Activation('relu'))
    
    # input subsampling layer (maxpooling) (2x2) for reduce data
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    
    #Input Flatten
    klasifikasi.add(Flatten())
    klasifikasi.add(Dense(150, activation = 'relu', init='he_normal'))
    klasifikasi.add(Dropout(0.5))
    klasifikasi.add(Dense(84, activation = 'relu', init='he_normal'))
    klasifikasi.add(Dropout(0.5))
    
    klasifikasi.add(Dense(jumlahKelas, activation='softmax',init='he_normal'))
    print("Full Connection Between Hidden Layers and Output Layers Completed")

    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(
            LOKASI_TRAINING,
            target_size=(64, 64),
            batch_size=20,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            LOKASI_TESTING,
            target_size=(64, 64),
            batch_size=20,
            class_mode='categorical')
    
    
    
    if isLoadedDariModel == True:
        namaFilenya = "modelKlasifikasi" + str(productionEpochnya) +".h5"
        if os.path.exists(namaFilenya) :
            klasifikasi = load_model(namaFilenya)
            datanya = klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        else:
            raise ValueError('Error: File Tidak Ada Harap Lakukan Training Terlebih Dahulu Sebelum Menggunakan Model')
    else:
        # compile CNN
        klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        print(klasifikasi.summary())
        print("Compiling Initiated")
       
        if MODENYA == 'production' :
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=hitungGambar(LOKASI_TRAINING),
                epochs=productionEpochnya,
                validation_data=test_set,
                validation_steps=hitungGambar(LOKASI_TESTING),
                callbacks=[tensorboard]
                )
            klasifikasi.save("modelKlasifikasi" + str(productionEpochnya) +".h5")
            
            test_steps_per_epoch = numpy.math.ceil(test_set.samples / test_set.batch_size )
    
            predictions = klasifikasi.predict_generator(test_set, steps=test_steps_per_epoch)
    
            predicted_classes = numpy.argmax(predictions, axis=1)
    
            true_classes = test_set.classes
            class_labels = list(test_set.class_indices.keys())
    
            report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
            print(report)
        else:
            print('Model Tidak Tersimpan')
        
        gambarHasilLatih()
        
    klasifikasi._make_predict_function()
    print("Compiling Completed")


def gambarHasilLatih():
    global datanya
    # Plot training & validation accuracy values
    plt.plot(datanya.history['acc'])
    plt.plot(datanya.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(datanya.history['loss'])
    plt.plot(datanya.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

        
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    global train_set, klasifikasi, IPNYA, PORTNYA, LOKASI_UPLOAD, PENYAKIT_TANAMAN
    print('-------------')
    print(request.method)
    print(request.files)
    print('-------------')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/' + os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            
            lokasiTest = LOKASI_UPLOAD + '/' + filename
          
            test_image = image.load_img('static/' + lokasiTest, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = klasifikasi.predict(test_image).tolist()
            '''result = pd.Series(result).to_json(orient='values')'''
            print(train_set.class_indices)
            '''return redirect(url_for('uploaded_file',filename=filename))'''
            print(result)
            
            hasil = {}
            dataJSON = {}
            allProba = {}
            loop = 0
            
            for cls, val in train_set.class_indices.items():
                '''hasil[cls] = result[0][train_set.class_indices[cls]]'''
                
                proba = result[0][train_set.class_indices[cls]]
                allProba[cls] = proba
                print(proba)
                if (proba > 0.0) and (proba <= 1.0) :
                    print('valnya : ' + str(val))
                    '''hasil.update({'datanya':{PENYAKIT_TANAMAN[val]},'probability':proba})'''
                    hasil["proba" + str(loop)] = PENYAKIT_TANAMAN[val]
                    hasil["proba" + str(loop)]['probability'] = proba
            
                    loop = loop + 1
            print(hasil)
            dataJSON['Debug'] = allProba
            dataJSON['penyakit'] = hasil
            dataJSON['uploadURI'] = 'http://' + IPNYA + ':' + str(PORTNYA) + url_for('static',filename=lokasiTest)
            
            return flask.jsonify(dataJSON)
        

    else:
        
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype='multipart/form-data'>
              <p><input type='file' name='file'>
                 <input type='submit' value='Upload'>
            </form>
            '''
  
        
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model_klasifikasi()
    app.run(host=IPNYA, port=PORTNYA,debug=True, use_reloader=False, threaded=False)