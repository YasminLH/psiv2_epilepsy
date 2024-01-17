
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F

import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import pickle

assert torch.cuda.is_available(), "GPU is not enabled"


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#variables:
Batch=512 #estandard es 512
epochs = 6 #estandard es 30  
#tipus="encoder"
tipus="lstm"
#divisio="pacient"
#divisio="window"
#divisio="interval"
divisio="recording"
limitador=450 #per limitar les dades(en interval i recording) estandard 450
limitador_2=2000 # estandard es 2000

#ir a annotated windows i pillar nombres split
#ir a metadades descomprimides i pillar los nombres de los parquets
#{"chb16":["npz","parquet"]}

Path_npz="/export/fhome/mapsiv/Epilepsy/annotated_windows"
Path_pd="/export/fhome/mapsiv03/parquets"

llista_npz=os.listdir(Path_npz)
llista_pd=os.listdir(Path_pd)

diccionari = {}

for arxiu in llista_npz:
    if ".zip" not in arxiu:
        nom_pacient = arxiu.split("_")[0]
        diccionari[nom_pacient]=[os.path.join(Path_npz,arxiu)]

for arxiu in llista_pd:
    nom_pacient = arxiu.split("_")[0]
    diccionari[nom_pacient]+=[os.path.join(Path_pd,arxiu)]

#por cada uno de ellos
#leer el npz i el parquet
#meter en lista tochisima ya limpiado


Y_train=[]
Y_test=[]
X_train=[]
X_test=[]

#torch.cat((x, x, x), 0)
def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
    
#(x,21,128)
print(diccionari)
if divisio=="pacient":
    n=0
    for key in tqdm(diccionari):
        arxiu_npz=diccionari[key][0]
        arxiu_parquet=diccionari[key][1]
        dataframe=pd.read_parquet(arxiu_parquet)
    
        if n<4:
            a=pd.Series(dataframe.index.tolist()).sample(limitador_2*2)
            classe=dataframe['class']
            classe_a=classe[a]
            Y_test.extend(list(classe_a))
            arxiunpz=np.load(arxiu_npz,allow_pickle=True)
            arxiu=arxiunpz['EEG_win']
            X_test.append(list(arxiu[a]))
            
            
        else:
            a=dataframe.index[dataframe['class'] == 1].tolist()[:limitador_2]
            b=dataframe.index[dataframe['class'] == 0].tolist()[:limitador_2]
            classe=dataframe['class']
            classe_a=classe[a]
            classe_b=classe[b]
            Y_train.extend(list(classe_a))
            Y_train.extend(list(classe_b))
            arxiunpz=np.load(arxiu_npz,allow_pickle=True)
            arxiu=arxiunpz['EEG_win']
            X_train.append(list(arxiu[a]))
            X_train.append(list(arxiu[b]))
    
        n+=1

    
    X_train=flatten_concatenation(X_train)
    X_test=flatten_concatenation(X_test)

elif divisio=="window":
    dades=[]
    Y=[]
    for key in tqdm(diccionari):
        #print(n)
    
    
        arxiu_npz=diccionari[key][0]
        arxiu_parquet=diccionari[key][1]
        
        dataframe=pd.read_parquet(arxiu_parquet)
        a=dataframe.index[dataframe['class'] == 1].tolist()[:limitador_2]
        b=dataframe.index[dataframe['class'] == 0].tolist()[:limitador_2]
        #print(len(dataframe))
        classe=dataframe['class']
        classe_a=classe[a]
        classe_b=classe[b]
        Y.extend(list(classe_a))
        Y.extend(list(classe_b))
    
    
        arxiunpz=np.load(arxiu_npz,allow_pickle=True)
        arxiu=arxiunpz['EEG_win']
        classe1=arxiu[a]
        classe2=arxiu[b]
        dades.append(list(classe1))
        dades.append(list(classe2))
    
    dades=flatten_concatenation(dades)
    X_train,X_test,Y_train,Y_test=train_test_split(dades,Y, test_size=0.3)
    
else:
    if divisio=="interval":
        separador='global_interval'
        extra=1
    elif divisio=="recording":
        separador='filename'
        extra=1.5
    else:
        raise AssertionError()
        
    x=0
    for key in tqdm(diccionari):

        arxiu_parquet=diccionari[key][1]
        dataframe=pd.read_parquet(arxiu_parquet)

        index_files=dataframe.groupby(by=separador).count().index
        total=len(index_files)
        a_agafar=(total//5) +1
        #part de test
        clau=index_files[0]
        A_test=pd.Series(np.where((dataframe[separador] == clau))[0])#aqui directe no cal remirar
        x=0
        for x in range(a_agafar-1):#per cada una de les de test
            clau=index_files[x+1]
            a=pd.Series(np.where((dataframe[separador] == clau))[0])
            A_test=pd.concat([A_test,a])

        x+=1

        if len(A_test)>limitador*a_agafar*extra:
            A_test=A_test.sample(int(limitador*a_agafar*extra),random_state=0)
        #part de train
        clau=index_files[x+1]


        de_tipus_1=dataframe.index[np.where((dataframe['class']==1) & (dataframe[separador] == clau))]
        total_de_test_tipus_1=len(de_tipus_1)

        if total_de_test_tipus_1>limitador:#per no pasar-nos
            total_de_test_tipus_1=limitador

        A_train=pd.Series(np.where((dataframe['class']==1) & (dataframe[separador] == clau))[0][:total_de_test_tipus_1])
        B_train=pd.Series(np.where((dataframe['class']==0) & (dataframe[separador] == clau))[0][:total_de_test_tipus_1])

        for clau in index_files[x+1:]:#per cada una de les de train
            x+=1
            clau=index_files[x]
            de_tipus_1=dataframe.index[np.where((dataframe['class']==1) & (dataframe[separador] == clau))]
            total_de_test_tipus_1=len(de_tipus_1)

            if total_de_test_tipus_1>limitador:#per no pasar-nos
                total_de_test_tipus_1=limitador

            a=pd.Series(np.where((dataframe['class']==1) & (dataframe[separador] == clau))[0][:total_de_test_tipus_1])
            b=pd.Series(np.where((dataframe['class']==0) & (dataframe[separador] == clau))[0][:total_de_test_tipus_1])

            A_train=pd.concat([A_train,a])
            B_train=pd.concat([B_train,b])
    
        #part de les xs
        arxiu_npz=diccionari[key][0]
        arxiunpz=np.load(arxiu_npz,allow_pickle=True)
        arxiu=arxiunpz['EEG_win']

        X_train.append(list(arxiu[A_train]))
        X_train.append(list(arxiu[B_train]))     
        X_test.append(list(arxiu[A_test]))
        
        
        #part de les ys
        classe=dataframe['class']
        classe_a_train=classe[A_train]
        classe_b_train=classe[B_train]
        classe_a_test=classe[A_test]
      
        Y_train.extend(list(classe_a_train))
        Y_train.extend(list(classe_b_train))
        Y_test.extend(list(classe_a_test))

    X_train=flatten_concatenation(X_train)
    X_test=flatten_concatenation(X_test)


#lista Y tocha
Y_train=torch.tensor(Y_train)
Y_train=F.one_hot(Y_train.long(),2)

Y_test=torch.tensor(Y_test)
Y_test=F.one_hot(Y_test.long(),2)
#lista dades tocha




print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))



#ni idea del maximo
#183W / 250W amb 128
#7010MiB / 12288MiB amb 160



class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (1,3), stride=1, padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=(1,2), padding=(0,1)),
            nn.Conv2d(128, 256, (1,3), stride=1, padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=(1,2), padding=(0,1)),
            nn.Conv2d(256, 512, (1,3), stride=1, padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=(1,2), padding=(0,1))
       
        )
        self.fusion = nn.Sequential(#nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(21,1,(1,1)),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d((512,4)),
                                    nn.Flatten()
                                    )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Capa de dropout
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
    
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.permute(0,2,1,3) 
        x = self.fusion(x)
        x = self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=8, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bias = False, dropout = 0.5)

        self.batch = nn.BatchNorm1d(num_features = self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    
    def forward(self, x,  h, c):
        
        
        x = x.permute(2,0,1)
        x, (h, c) = self.lstm(x, (h, c))
        x = self.batch(x[-1])# sino va probar con x[-1]
        x = self.linear(x)
        
        return x, h, c
        
        
    def init_hidden(self, batch_size):
        " Initialize the hidden state of the RNN to zeros"
        return nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim)), nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        
    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

def train(model, loader, optimizer, criterion, device, tipus):
    loss = 0
    model.train()
    losses =list()

    for batch_features,target in loader:
    
        #print(batch_features.shape)
        #batch_features = batch_features.unsqueeze(1)
        #print(batch_features.shape)
        
        batch_features = batch_features.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        if tipus=="lstm":
            h,c = model.init_hidden(len(target))
            h=h.to(device)
            c=c.to(device)
            outputs, h, c = model(batch_features, h,c)
        elif tipus=="encoder":
            outputs = model(batch_features)
        else:
            raise AssertionError()
        #h.detach()
        #c.detach()
        train_loss = criterion(outputs, target)
      
        train_loss.backward()#retain_graph=True)
       
        optimizer.step()

        loss += train_loss.item()
        

    
    loss = loss / len(loader)
    print(f"Train loss = {loss:.6f}")
    losses.append(loss)
       
    return loss

def test(model, loader, criterion, device, tipus):
    loss = 0
    model.eval()
        
    for batch_features,Y in loader:
    
        #batch_features = batch_features.unsqueeze(1)
        batch_features = batch_features.to(device)

        with torch.no_grad():
            if tipus=="lstm":
                h,c = model.init_hidden(len(Y))
                h=h.to(device)
                c=c.to(device)
                outputs, h, c = model(batch_features, h,c)
            elif tipus=="encoder":
                outputs = model(batch_features)
            else:
                raise AssertionError()

        Y= Y.to(device)
        test_loss = criterion(outputs, Y)
        
        loss += test_loss.item()


    loss = loss / len(loader)
    print(f"Test loss = {loss:.6f}")
    return loss
    
    
os.makedirs('pickles', exist_ok=True) 
os.makedirs('modelos', exist_ok=True) 
os.makedirs('tensors', exist_ok=True) 

#de train hacer kfold
kf = KFold(n_splits=4)
#para cada fold entrenar un modelo



criterion = nn.BCEWithLogitsLoss()


ultimos_valores=[]



for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
    Xnp = np.array(X_train)
    #Ynp = np.array(Y_train)

    training=[]
    for dada1,dada2 in zip(Xnp[train_index],Y_train[train_index]):
        training.append((np.array(dada1,dtype=np.float32),np.array(dada2,dtype=np.float32)))

    loader_train= torch.utils.data.DataLoader( training, batch_size=Batch, shuffle=True )

    testing=[]
    for dada1,dada2 in zip(Xnp[test_index],Y_train[test_index]):
        testing.append((np.array(dada1,dtype=np.float32),np.array(dada2,dtype=np.float32)))

    loader_test= torch.utils.data.DataLoader( testing, batch_size=Batch, shuffle=True )
    
    
    losses=[]
    losses_test=[]
    model = ConvAE()
    if tipus=="encoder":
        model = ConvAE()
    elif tipus=="lstm":
        #torch.autograd.set_detect_anomaly(True)
        model = LSTM(input_dim= 21,hidden_dim= 20, batch_size=3, output_dim=2, num_layers=2)
    else:
        raise AssertionError()

    model.to(device)
    
    optimizer = torch.optim.Adamax(model.parameters(),  lr=0.001 )
    
    
    print("epocas")

    
    
    for epoch in tqdm(range(epochs)):
    
        loss=train(model, loader_train, optimizer, criterion, device, tipus)
        losses.append(loss)

        loss_test=test(model, loader_test, criterion, device, tipus)
        losses_test.append(loss_test)
    
    print(f"Modelo {i}: Pérdidas en train: {losses}")
    print(f"Modelo {i}: Pérdidas en test: {losses_test}")

    ultimos_valores.append(losses_test[-1])
    nom = str(i)
    torch.save(model,"modelos/modelos_"+tipus+"_"+nom+"_"+divisio)
    data = (losses, losses_test)
    
    
    file_path = 'pickles/loses_pickle_'+tipus+nom+divisio
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


#mirar que modelo generaliza mejor
#ir a pillar ese modelo
index=np.argmin(ultimos_valores)

model_final = torch.load("modelos/modelos_"+tipus+"_"+str(index)+"_"+divisio)
print(model_final)
model_final = model_final.to(device)
print(model_final)
print("\n")
print("modelos/modelos_"+tipus+"_"+str(index)+"_"+divisio)



def validation(model, loader, criterion, device, tipus):
    loss = 0
    model.eval()
    sortides = torch.Tensor().to(device)
    veritats = torch.Tensor().to(device)

        
    for batch_features,Y in loader:
    
        #batch_features = batch_features.unsqueeze(1)
        batch_features = batch_features.to(device)

        with torch.no_grad():
            if tipus=="lstm":
                h,c = model.init_hidden(len(Y))
                h=h.to(device)
                c=c.to(device)
                outputs, h, c = model(batch_features, h,c)
            elif tipus=="encoder":
                outputs = model(batch_features)
            else:
                raise AssertionError()

        
        Y= Y.to(device)
        test_loss = criterion(outputs, Y)
        
        sortides = torch.cat((sortides, outputs), 0)
        veritats = torch.cat((veritats, Y), 0)
        
        
        loss += test_loss.item()

    
    
    loss = loss / len(loader)

    print(loss)
    
    torch.save(sortides, 'tensors/sortides_'+tipus+"_"+divisio)
    torch.save(veritats, 'tensors/veritats_'+tipus+"_"+divisio)


Xnp = np.array(X_test)
#Ynp = np.array(Y_train)

validating=[]
for dada1,dada2 in zip(Xnp,Y_test):
    validating.append((np.array(dada1,dtype=np.float32),np.array(dada2,dtype=np.float32)))
    
loader_validation= torch.utils.data.DataLoader( validating, batch_size=Batch, shuffle=True )

validation(model_final, loader_validation, criterion, device, tipus)



