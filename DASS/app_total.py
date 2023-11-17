import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io.wavfile
import os
import seaborn as sns
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


path_file = os.listdir('data/04-11-23')
label = [filesname.replace('.csv', '') for filesname in path_file]

# label
file = {}
for no_label in label:
    data = pd.read_csv(f'data/04-11-23_edit/{no_label}.csv',skiprows=2)
    file[no_label] = data
    globals()[no_label] = data


window_size = 6000
window_step = 1800


file_segmen = []
for no_segmen in range(0,len(label)):
    file[label[no_segmen]].columns = ["num","ir","waktu"]
    path1 = file[label[no_segmen]][(file[label[no_segmen]].waktu >10)]
    path1 = path1[(path1.waktu <=190)]
    
    for no_window in range(0,len(path1),window_step):
        window = path1[no_window:no_window+window_size]
        file_segmen.append(window)


info = []
minmaxx = []
signals = []
data_var = []
data_std = []
ppg_elgendi = []
for no_minmax in range(0,len(file_segmen)):
    minmax = MinMaxScaler(feature_range=(0,1))
    
    path_minmax1 = minmax.fit_transform(file_segmen[no_minmax])
    path_minmax_seg1 = pd.DataFrame(path_minmax1)[1]
    path_minmax_seg1 = nk.ppg_clean(path_minmax_seg1, sampling_rate=100, method='elgendi')
    ppg_elgendi.append(path_minmax_seg1)
    path_minmax_seg1 = pd.DataFrame(path_minmax_seg1)

    signals1, info1 = nk.ppg_process(path_minmax_seg1, sampling_rate=100)

    minmaxx.append(path_minmax_seg1)
    signals.append(signals1)
    info.append(info1)

    data_var.append(float(pd.DataFrame(info1['PPG_Peaks']).diff().var()))
    data_std.append(float(pd.DataFrame(info1['PPG_Peaks']).diff().std()))


analyze_signals = []
bpm = []
hrv = []
rmssd = []
sdnn = []
for no_analyze in range(0,len(file_segmen)):
    analyze_signals1 = nk.ppg_analyze(signals[no_analyze], sampling_rate=100)

    analyze_signals.append(analyze_signals1)
    bpm.append(float(analyze_signals1['PPG_Rate_Mean']))
    hrv.append(float(analyze_signals1['HRV_MeanNN']))
    rmssd.append(float(analyze_signals1['HRV_RMSSD']))
    sdnn.append(float(analyze_signals1['HRV_SDNN']))
    
sistol = []
diastol = []
distance = []
peak_sistol = {}
peak_diastol = {}

for no_sisdis in range(0,len(file_segmen)):  
    path = ppg_elgendi[no_sisdis]
    med = path.max()/4
    sistol1 = []
    for i in range(1, len(path) - 1):
        if path[i] > path[i - 1] and path[i] > path[i + 1] and path[i] > med:
            sistol1.append(i)


    diastol1 = []
    for i in range(1, len(path) - 1):
        if path[i] > path[i - 1] and path[i] > path[i + 1] and path[i] < med:
            diastol1.append(i)

    peak_sistol[no_sisdis] = sistol1
    peak_diastol[no_sisdis] = diastol1
    sistol.append(float(pd.DataFrame(path[sistol1]).mean()))
    diastol.append(float(pd.DataFrame(path[diastol1]).mean()))
    distance.append(float(pd.DataFrame(path[sistol1]).mean())-float(pd.DataFrame(path[diastol1]).mean()))

    # plt.plot(path);
    # plt.plot(diastol1, path[diastol1], "x");
    # plt.plot(sistol1, path[sistol1], "x");

dataajah = {'BPM':bpm,'HRV':hrv,'RMSSD':rmssd,'SDNN':sdnn,'VAR':data_var,'STD':data_std,'SISTOL':sistol,'DIASTOL':diastol,'DISTANCE':distance}
dataaaaa = pd.DataFrame(dataajah)

bahan_label = pd.read_excel('excel/label_14-11-23.xlsx')
nama = bahan_label['nama'] 
kecemasan = bahan_label['kecemasan'] 
angka_kecemasan = bahan_label['angka_kecemasan']

named = []
anxd = []
anganxd = []
for name,anx,anganx in zip(nama,kecemasan,angka_kecemasan):
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)
    named.append(name)

    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)
    anxd.append(anx)

    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)
    anganxd.append(anganx)


anxd = pd.DataFrame(anxd)
anxd.column = 'KECEMASAN'
anganxd = pd.DataFrame(anganxd)
anganxd.column = 'KECEMASAN REGRESI'
named = pd.DataFrame(named)
named.column = 'NAMA'


data_real = pd.concat([named,dataaaaa,anxd,anganxd], axis=True)
data_real.columns = [
'NAMA',
'BPM',
'HRV',
'RMSSD',
'SDNN',
'VAR',
'STD',
'SISTOL',
'DIASTOL',
'DISTANCE',
'KECEMASAN',
'KECEMASAN_REGRESI']


minmax = MinMaxScaler()
sm = SMOTE(random_state=30)
le = LabelEncoder()
model = load('knn_model.joblib')
array = ['BPM','HRV','RMSSD','SDNN','VAR','STD','SISTOL','DIASTOL','DISTANCE']
# data = pd.read_excel('excel/bahan_knn_new/DATA_KNN_6000x1800_NEW.xlsx')
x_rmsdd = data_real[array]
y_klasifikasi = data_real['KECEMASAN']
x_rmsdd = minmax.fit_transform(x_rmsdd)
x_rmsdd_smote,y_rmsdd_smote = sm.fit_resample(x_rmsdd,y_klasifikasi)
x_rmsdd_smote_train, x_rmsdd_smote_test, y_rmsdd_smote_train, y_rmsdd_smote_test = train_test_split(x_rmsdd_smote, y_rmsdd_smote, test_size= 0.2, random_state=20)



alat_test = pd.read_excel('alat_test.xlsx')
angka = 0
tambah = len(alat_test)
alat_test = alat_test[angka:angka+tambah]
x_real = alat_test[array]
x_real_minmax = minmax.fit_transform(x_real)
y_real = alat_test['KECEMASAN']

# akurasi = []
# for i in range(1,51):
#     k = i
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(x_rmsdd_smote_train,y_rmsdd_smote_train)
#     akurasi.append((model.score(x_real_minmax,y_real))*100)
#     # predict = model.predict(x_real_minmax)

# akurasi = pd.DataFrame(akurasi)
# akurasi[0].nlargest(5)

print(f'akurasinya adalah : {(model.score(x_real_minmax,y_real))*100}%')
