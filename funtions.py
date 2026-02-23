from distutils.log import error
from re import X
import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import sqlite3
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import torch.nn.init as init
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable 

from utils import *
from Myfolds import *

# Chama a gpu cuda disponível.Caso não tenha gpu disponível , usa a cpu

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# Arquitetura da rede FC
class LSTM_Network(nn.Module):

    def __init__(self,input_size,output_size,hidden_size,dropout_value,num_layers,dropout_value_lstm,option_bidirectional,use_batchnorm=False):
        super(LSTM_Network, self).__init__() 

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_value = dropout_value
        self.num_layers = num_layers
        self.dropout_value_lstm = dropout_value_lstm
        self.bidirectional = option_bidirectional
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(input_size)
        
        if self.bidirectional == True:
           self.num_directions = 2
        else:
            self.num_directions = 1 

        self.lstm = nn.LSTM(self.input_size,  #512 -> 
                            self.hidden_size, # 128 ->
                            self.num_layers, # ?? 
                            batch_first = True,  ### if true ((batch, seq, feature))
                            #=self.dropout_value_lstm,
                            bidirectional=self.bidirectional) 
        self.dropout_lstm_layer = nn.Dropout(self.dropout_value_lstm)  
        self.dense_hidden = nn.Linear(self.hidden_size, self.hidden_size) #128 --> 128
        self.dropout_linear_layer = nn.Dropout(self.dropout_value) #(128 -> 128)
        self.dense = nn.Linear(self.hidden_size, self.output_size) #128 -> 1
        

        # inicialização de pesos por fora do default
        #self.initialize_weights_lstm()
        #self.init_weights_denseh()


    def initialize_weights_lstm(self):
        print('INICIALIZANDO PESOS: ')
        
        init.kaiming_uniform_(self.lstm.weight_ih_l0, nonlinearity='relu')
        init.kaiming_uniform_(self.lstm.weight_hh_l0, nonlinearity='relu')
        init.constant_(self.lstm.bias_ih_l0, 0)
        init.constant_(self.lstm.bias_hh_l0, 0)
        
        # Inicialização dos pesos da camada densa
        init.xavier_uniform_(self.dense_hidden.weight)
        init.constant_(self.dense_hidden.bias, 0)
        
        init.xavier_uniform_(self.dense.weight)
        init.constant_(self.dense.bias, 0)

    def init_weights_denseh(self):
        # Inicialização dos pesos com VarianceScaling
        print('INICIALIZANDO PESOS DAS CAMADAS DENSAS:')
        # nn.init.uniform_(self.dense.weight, -1.0 / self.input_size, 1.0 / self.input_size)
        # Inicialização dos viéses com zeros
        #nn.init.zeros_(self.dense_hidden.bias)
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(self.dense.weight, -1.0 / self.input_size, 1.0 / self.input_size)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.use_batchnorm:
            x = x.permute(0, 2, 1)  # (batch, timesteps, features) → (batch, features, timesteps)
            x = self.batchnorm(x)
            x = x.permute(0, 2, 1)  # Volta para (batch, timesteps, features)

        
        
        x = self.dropout_lstm_layer(x)

        # Inicializando os estados ocultos com tensores preenchidos por 0  
        h_0 = torch.zeros(self.num_layers*self.num_directions, x.size(0), self.hidden_size, dtype=torch.float32) # hidden state    --> 1,32,512
        c_0 = torch.zeros(self.num_layers*self.num_directions, x.size(0), self.hidden_size, dtype=torch.float32)
        h_0,c_0 = h_0.to(device),c_0.to(device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # shape de hn -> [1,32,128]
     
        output = output[:, -1, :]
        output = self.dense_hidden(output)
        output = torch.tanh(output)        
        output = self.dropout_linear_layer(output)
    
        output = self.dense(output)
        return output     
    
    


# Dataset Lstm
class LSTM_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,fold_number,mode,overlap,causal,n_steps):
        '''Define os valores iniciais.'''
        self.mode = mode
        
        self.fold_number = fold_number
            
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []
        self.overlap = overlap
        print(self.fold_number)
        
        ## Serve Apenas para o 10F 
        
        if name_Folds == '10F':
            number_of_frames_train_filename = CONST_STR_DATASET_FOLDS_DATAPATH+'fold_'+str(self.fold_number)+"_train_numberofframes"+".npy"
            number_of_frames_test_filename = CONST_STR_DATASET_FOLDS_DATAPATH+'fold_'+str(self.fold_number)+"_test_numberofframes"+".npy"
        
        ## Como fazer para o 308 344:
        #[NOTE] Criei dois NOF:
        elif( name_Folds == '308') or ( name_Folds == '344'):
            
            number_of_frames_train_filename = NOF_308_344 +"nof_train_"+name_Folds+".npy"
            number_of_frames_test_filename = NOF_308_344 +"nof_validation_"+name_Folds+".npy"
        
        elif (name_Folds == 'f4'):
            number_of_frames_train_filename = CONST_STR_DATASET_FOLDS_DATAPATH+'fold_4'+"_train_numberofframes.npy"
            number_of_frames_test_filename = CONST_STR_DATASET_FOLDS_DATAPATH+'fold_4'+"_test_numberofframes"+".npy"
 
        try:
            training_nof = np.load(number_of_frames_train_filename)
            testing_nof = np.load(number_of_frames_test_filename)
        except:
            print("Could not open one or more of the following files:")
            print("\t"+number_of_frames_train_filename)
            print("\t"+number_of_frames_test_filename)
            exit(1)
        
        for video_index in folds[self.fold_number][self.m]:
            if self.mode == 'val':
                mode_mtl = 'test'
            elif self.mode == 'train':
                mode_mtl = 'train'
            else:
                print("Modo não encontrado")
                exit(1)
            # Arquivos do felipe
            if (FEATURES == 'Felipe'):
                training_frames = np.load(os.path.join(PATH_FEATURES_FELIPE + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1) 
            if (FEATURES == 'MatheusNoPIL'):
                training_frames = np.load(os.path.join(PATH_FEATURES_MATHEUSNOPIL + video_index +'_22_09_2024_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1) 
            
 
            # Arquivo para o uso dos targets do matheus 
            #[NOTE: Isso não parece certo...]
            if (FEATURES == 'Matheus'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1) 

            #Features que eu fiz utilizando as estatisticas variaveis e folds inteiros
            if (FEATURES == 'MatheusNaotoF'):
                
                # isso é um JEITINHO deve ter maneira mais otimizadas de fazer para usar apenas o fold4 do 10F
                if (name_Folds == 'f4'):
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_4_'+mode_mtl+'_output_data.npy')
                else:
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_'+str(self.fold_number)+'_'+mode_mtl+'_output_data.npy')
                
                    
            

            if (FEATURES == 'MatheusNaoto'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)

            if (FEATURES == 'test_features'):
                training_frames = np.load(os.path.join(PATH_FEATURES_TEST_VIDEOS + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_DATA_TO_EXTRACTION + video_index + '/audioData.npy'))
                training_targets = np.mean(training_targets, axis=1)

            if (FEATURES == 'Matheus') or (FEATURES == 'MatheusNaoto') or (FEATURES == 'MatheusNaotoF') :
                pass
            else:
                self.frames_list.append(training_frames) ##Desmarcar QUANDO NAO É MATHEUS_LIMA - ARRUMAR
            self.pressures_list.append(training_targets)
        
        for i in self.frames_list:
            for j in range(i.shape[0]):
                self.list_of_all_frames.append(i[j])
        
        for i in self.pressures_list:
            for j in range(i.shape[0]):
                self.list_of_all_pressures.append(i[j])
         
        self.frames_array = np.array(self.list_of_all_frames)
        self.pressures_array = np.array(self.list_of_all_pressures)

        
        if (FEATURES == 'Matheus'):
            # features-train do matheus
            path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            if self.mode == 'train':
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_train_input_data_gap.npy')
                
                
            else:
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_test_input_data_gap.npy')
            
            
            self.frames_array = np.array(training_frames)
            if (FEATURES == 'Matheus'):
                self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            else:
                pass
        elif (FEATURES == 'MatheusNaoto'):
            if self.mode == 'train':
                training_frames = np.load(PATH_FEATURES_MATHEUSN+'fold'+ str(self.fold_number)+'_train_features_matheusnaoto.npy')
            else:
                training_frames = np.load(PATH_FEATURES_MATHEUSN+'fold'+ str(self.fold_number)+'_val_features_matheusnaoto.npy')
            self.frames_array = np.array(training_frames)
            if (FEATURES == 'MatheusNaoto'):
                self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            else:
                pass 
            
        elif (FEATURES == 'MatheusNaotoF'):
            if self.mode == 'train':
                if (name_Folds == 'f4'):
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_4_train_features.npy')
                else:
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_'+ str(self.fold_number)+'_train_features.npy')
                
            else:
                if (name_Folds == 'f4'):
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_4_val_features.npy')
                else:
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_'+ str(self.fold_number)+'_val_features.npy')


            self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            
        else:
            self.frames_array = torch.from_numpy(self.frames_array).float()
            mean = torch.mean(self.frames_array)
            std = torch.std(self.frames_array)
            print ('MEDIA 308: ',mean, '... STD 308: ', std)

        
        
        
        self.pressures_array = torch.from_numpy(self.pressures_array).float()
        if (FEATURES == 'Matheus') or (FEATURES == 'MatheusNaoto'):
            self.data_len = len(self.frames_array)#USAR PARA FRAMES DO MATHEUS_LIMA
        else:
            self.data_len = len(self.list_of_all_frames) ##USAR PARA FEATURES DO FELIPE
          
        

        #self.data_len = len(self.list_of_all_pressures) ##Não to usando
        #self.data_len = len(self.list_of_all_frames) 

        if (overlap == True) and (causal == True): 
         

            X = []
            y = []

            # Com sobreposição de janelas e de forma causal
            
            for i in range(self.data_len):

                end_ix = i + n_steps
            
                if end_ix > len(self.pressures_array)-1:
                    break

                seq_x, seq_y = self.frames_array[i:end_ix], self.pressures_array[end_ix-1]
    
                X.append(seq_x)
                y.append(seq_y)

        if (overlap == True) and (causal == False):
            
            
            X = []
            y = []
            
            target_size = int((n_steps)/2)
            
            if self.mode == "train":
                frame_sum = 0   # This variable keeps track of what frame in testing_images is being processed now
                
                for i in range(len(training_nof)):  # For each video in testing_images . . .
                    # print('training_nof_sum'+str(np.sum(training_nof)))
                    start_index = frame_sum+n_steps
                    
                    # print(len(self.frames_array))
                    end_index = frame_sum+training_nof[i]

                    
                    for j in range(start_index, end_index):     # For each window in this video . . .
                        indices = range(j-n_steps, j)
                        
                        
                        X.append(np.reshape(self.frames_array[indices], (n_steps, 512))) ##512 que é a saida da VGG16 (7,7,512)
                        y.append(self.pressures_array[j-target_size])
                        #print(j-target_size)

                    frame_sum += training_nof[i]
                
                    #print('frames_Sum: ', frame_sum)
                #print(end_index)

            elif self.mode == 'val':
                frame_sum = 0   # This variable keeps track of what frame in testing_images is being processed now
                for i in range(len(testing_nof)):  # For each video in testing_images . . .
                    
                    
                    start_index = frame_sum+n_steps
                    end_index = frame_sum+testing_nof[i]
                    
                    for j in range(start_index, end_index):     # For each window in this video . . .
                        indices = range(j-n_steps, j)
                        
                        #print(i)
                        
                        X.append(np.reshape(self.frames_array[indices], (n_steps, 512))) ##512 que é a saida da VGG16 (7,7,512)
                        y.append(self.pressures_array[j-target_size])
                    
                    frame_sum += testing_nof[i]
                #print(end_index)
            
            
            
            
            
            # for i in range(self.data_len):
                
            #     end_ix = i + n_steps
            
            #     if end_ix >= (self.data_len):
            #         print('end_ix: ',end_ix)
            #         break
            #     #print('intervalo', i,',',end_ix)
            #     seq_x = self.frames_array[i:end_ix]

            #     target_index = i + (n_steps//2 ) -1
            #     #(n_steps//2 ) -> 16
            #     seq_y = self.pressures_array[target_index]
            #     """
            #     w_frames_file = open('windows_with_overlap_without_causal.txt','a')
            #     w_frames_file.write('\n i {} end_ix {} - len seq_frames {}\n'.format(i,end_ix,len(seq_x)))
            #     w_frames_file.write(' {}\n'.format(seq_x))
            #     w_frames_file.write('target= {}\n'.format(seq_y))
            #     w_frames_file.close()
            #     """
                
            #     X.append(seq_x)
            #     y.append(seq_y)
            
        
        self.frames_array = X
        
        self.pressures_array = y
        self.len = len(self.frames_array)
        #print('NUMBER OF BATCHES:   ',self.len)
        # print('DATASE_x', self.m, X)
        # print('DATASE_y', self.m, y)
         

    def __getitem__(self, index): # indice do janela escolhida e do target correspondente a janela
        '''Retorna o item de número determinado pelo indice'''
        
        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''
        
        # Retorno o numero de janelas
        return self.len
    

    


# Arquitetura da rede FC para a VGG16
class VGG_Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers_size_list,dropout_value):
        super(VGG_Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_size_list = hidden_layers_size_list
        size_current = input_size
        self.layers = nn.ModuleList()
        for size_index in hidden_layers_size_list:
            self.layers.append(nn.Linear(size_current, size_index))
            size_current = size_index
        self.layers.append(nn.Linear(size_current, output_size))

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        for layer in self.layers[:-1]: # Estou pegando todas as camadas,exceto a última 
            x = torch.tanh(layer(x))
        x = self.dropout(x)
        x = self.layers[-1](x)
        return x   
      

def initialize_weights(model):
    print('INICIALIZANDO PESOS: ')
    
    
    for layer in model.layers:
        nn.init.normal_(layer.weight)
        if layer.bias is not None:
            nn.init.normal_(layer.bias)
        

# Dataset  que retorna a tupla (frames,pressoes) de 1 fold
class VGG_Dataset(Dataset):
    '''Classe que representa nosso dataset. Deve herdar da classe Dataset, em torch.utils.data
    '''

    def __init__(self, folds,fold_number,mode):
        '''Define os valores iniciais.'''
        self.mode = mode
        self.fold_number = fold_number
        self.frames_list = []
        self.pressures_list = []
        self.list_of_all_frames = []
        self.list_of_all_pressures = []
        
        

        for video_index in folds[self.fold_number][self.mode]:
            if self.mode == 'val':
                mode_mtl = 'test'
            elif self.mode == 'train':
                mode_mtl = 'train'
            else:
                print("Modo não encontrado")
                exit(1)
            # Meus arquivos de features     
            #training_frames = np.load(os.path.join(PATH_FEATURES_CAROL,'Features_'+video_index+'.npy'))
            #training_targets = np.load(os.path.join(PATH_FEATURES_CAROL,'Sound-Pressures_'+ video_index +'.npy'))
            
            # Arquivos do felipe 
            if (FEATURES == 'Felipe'):
                training_frames = np.load(os.path.join(PATH_FEATURES_FELIPE + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)

            if (FEATURES == 'MatheusNoPIL'):
                training_frames = np.load(os.path.join(PATH_FEATURES_MATHEUSNOPIL + video_index +'_22_09_2024_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1) 
            
            # Arquivo para o uso dos targets do matheus
            if (FEATURES == 'Matheus'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)

            # Arquivos para as features extraidas carregando o modelo em pytorch e usando d vgg do tf/keras
            if (FEATURES == 'torch_model_with_weights_of_tf/keras'):
                training_frames = np.load(os.path.join(PATH_FEATURES_TF_KERAS + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE + video_index +'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)

            # Arquivos para as features extraidas dos 10 novos vídeos de teste.
            if (FEATURES == 'test_features'):
                training_frames = np.load(os.path.join(PATH_FEATURES_TEST_VIDEOS + video_index +'_features.npy'))
                training_targets = np.load(os.path.join(PATH_DATA_TO_EXTRACTION + video_index + '/audioData.npy'))
                training_targets = np.mean(training_targets, axis=1)
           

            if (FEATURES == 'MatheusNaotoF'):
                
                # isso é um JEITINHO deve ter maneira mais otimizadas de fazer para usar apenas o fold4 do 10F
                if (name_Folds == 'f4'):
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_4_'+mode_mtl+'_output_data.npy')
                else:
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_'+str(self.fold_number)+'_'+mode_mtl+'_output_data.npy')

            if (FEATURES == 'Matheus') or (FEATURES == 'MatheusNaoto') or (FEATURES == 'MatheusNaotoF'):
                pass
            else:
                self.frames_list.append(training_frames) ##Desmarcar QUANDO NAO É MATHEUS_LIMA - ARRUMAR
            self.pressures_list.append(training_targets)
            
            
            
        for video_index in self.frames_list:
            for frame_index in range(video_index.shape[0]):
                self.list_of_all_frames.append(video_index[frame_index])
        
        for video_index in self.pressures_list:
            for pressure_index in range(video_index.shape[0]):
                self.list_of_all_pressures.append(video_index[pressure_index])
         
        self.frames_array = np.array(self.list_of_all_frames)
        self.pressures_array = np.array(self.list_of_all_pressures)

        if (FEATURES == 'Matheus'):
            # features-train do matheus
            path_matheus = '/home/mathlima/dataset/folds/vgg16/'
            if self.mode == 'train':
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_train_input_data_gap.npy')
            else:
                training_frames = np.load(path_matheus+'fold_'+ str(self.fold_number)+'_test_input_data_gap.npy')
            #print(training_frames)
            self.frames_array = np.array(training_frames)
            if (FEATURES == 'Matheus'):
                self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            else:
                pass



        elif (FEATURES == 'MatheusNaoto'):
            if self.mode == 'train':
                training_frames = np.load(PATH_FEATURES_MATHEUSN+'fold'+ str(self.fold_number)+'_train_features_matheusnaoto.npy')
            else:
                training_frames = np.load(PATH_FEATURES_MATHEUSN+'fold'+ str(self.fold_number)+'_val_features_matheusnaoto.npy')
            self.frames_array = np.array(training_frames)
            if (FEATURES == 'MatheusNaoto'):
                self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            else:
                pass 


        elif (FEATURES == 'MatheusNaotoF'):
            if self.mode == 'train':
                if (name_Folds == 'f4'):
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_4_train_features.npy')
                else:
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_'+ str(self.fold_number)+'_train_features.npy')
                
            else:
                if (name_Folds == 'f4'):
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_4_val_features.npy')
                else:
                    training_frames = np.load(PATH_FEATURES_MATHEUSNAOTOF+'Fold_'+ str(self.fold_number)+'_val_features.npy')
           
            self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!


        
        else:
            self.frames_array = torch.from_numpy(self.frames_array).float()
            mean = torch.mean(self.frames_array)
            std = torch.std(self.frames_array)
            print ('MEDIA 308: ',mean, '... STD 308: ', std)


        
        self.pressures_array = torch.from_numpy(self.pressures_array).float()
        if (FEATURES == 'Matheus') or (FEATURES == 'MatheusNaoto') or (FEATURES == 'MatheusNaotoF'):
            self.data_len = len(self.frames_array)#USAR PARA FRAMES DO MATHEUS_LIMA
        
        else:
            self.data_len = len(self.list_of_all_frames)  ##USAR PARA FEATURES DO FELIPE
        #self.data_len = len(self.frames_array)  #USAR PARA FRAMES DO MATHEUS_LIMA


    def __getitem__(self, index): # indice do fold escolhido e modo de treino ou validação
        '''Retorna o item de número determinado pelo indice'''
        return self.frames_array[index],self.pressures_array[index]

    def __len__(self):
        '''Número total de amostras'''

        return self.data_len

# Função de treino 
def train(model,train_dataset,loss_function,optimizer,batch_grid):
    model.train(True)
    train_loss = 0.0

    
    train_loader = DataLoader(train_dataset,batch_size=batch_grid,shuffle=OPTION_SHUFFLE,num_workers=OPTION_NUM_WORKERS)
    
    for i,data in enumerate(train_loader): 
        optimizer.zero_grad()
        #frames,pressure in train_loader 
        frames, pressure = data
        #np.save('scratch/frames_lstm.npy', frames)
        #np.save('scratch/pressures_lstm.npy', pressure)
        frames, pressure = frames.to(device), pressure.to(device)
        pressure_aux = pressure 
        pressures = pressure_aux[:,None]
        
        #print(pressures.size())
        
        #torch.save(model.state_dict(), checkpoint_path00)
        pred = model(frames)
                
        #pred_vgg = pred.cpu()
        #array_np = pred_vgg.detach().numpy()
        #np.save('scratch/predicoes_lstm.npy', array_np)

        
        #torch.save(model.state_dict(), checkpoint_path0)

        
        loss = loss_function(pred,pressures)
        loss.backward()
        
        #torch.save(model.state_dict(), checkpoint_path1)

        optimizer.step() #self.frames_array = torch.from_numpy(training_frames).float() # isso para as features do matheus!
            #print(self.frames_array)
            #print(np.shape(self.frames_array))
        #torch.save(model.state_dict(), checkpoint_path2)

        
        train_loss += frames.size(0) * loss.item()
        
    return (train_loss/len(train_dataset))
    
# Função de validação 
def validation(model,val_dataset,loss_function,path,fold_index,batch_grid):
    model.eval()
    val_loss = 0.0
    min_val_loss = 1_000_000
    running_loss =0 #variavel 

    val_loader = DataLoader(val_dataset,batch_size=batch_grid,shuffle=False,num_workers=OPTION_NUM_WORKERS)
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            frames, pressure = data
            frames, pressure = frames.to(device), pressure.to(device)
            
            pressure_aux = pressure
            pressures = pressure_aux[:,None]
            
            # Passando os dados para o modelo para obter as predições
            pred = model(frames)

            # Calculando a perda através da função de perda
            
            loss = loss_function(pred,pressures)
            #print('VALIDATION LOSS per batch: ',loss)
            val_loss += frames.size(0) * loss.item()
            running_loss += loss
            
            if min_val_loss > val_loss:
                #print(f'Validation Loss Decreased({min_val_loss:.6f}-->{val_loss:.6f}) \t Saving The Model')
                min_val_loss = val_loss
                
                best_model_saved_path = os.path.join(path,'best_model/')

                if not os.path.exists(best_model_saved_path):
                    os.makedirs(best_model_saved_path)

                # Salvando o modelo 
                torch.save(model.state_dict(),best_model_saved_path+'best_model_fold_'+str(fold_index)+'.pth') 
        #print(len(val_dataset))
        return (val_loss/len(val_dataset))

# Função que retorna o otimizador 
def optimizer_config(opt,model,value_lr):
    if (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr = value_lr)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
                # print(f"Parameter: {name}")
                # print(f" - Shape: {param.shape}")
                # print(f" - Value: {param.data}")
                

    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr = value_lr)
    if (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr = value_lr)
    return optimizer

# Função que plota gráfico da curva de treino 
def graphic_of_training(df_train,df_val,fold_index,path,epochs_number):
    df_train.plot(ax=plt.gca())
    df_val.plot(ax=plt.gca())
    limiar = 4 #train_loss_max*0.25
    plt.axis([0,(epochs_number+1),0,limiar])
    plt.title('Loss_plot-Fold_'+str(fold_index))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+'Loss_plot-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que plota gráfico com a curva de treino de todos os folds 
def graphic_of_training_all_folds(df_train_all_folds,df_val_all_folds,path,epochs_grid):
    mean_val = pd.DataFrame(columns=['val_loss_mean'])
    mean_train = pd.DataFrame(columns=['train_loss_mean'])

    

    mean_val['val_loss_mean'] = df_val_all_folds.mean(axis=1)
    mean_train['train_loss_mean'] = df_train_all_folds.mean(axis=1)

    mean_val_log = np.log(mean_val)
    mean_train_log = np.log(mean_train)

    #Só adicionei o _log
    mean_val_log.plot(color='blue',alpha=1.0,ax=plt.gca(),legend=False)
    mean_train_log.plot(color='pink',alpha=1.0,ax=plt.gca(),legend=False)
    
    # iterando entre as colunas do data frame de validação
    for column in df_val_all_folds:
        df_column = pd.DataFrame(df_val_all_folds[column].values)
        df_column.plot(color='blue',alpha=0.4,ax=plt.gca(),legend=False)

    plt.title('Loss_plot-All_Folds')
    limiar = 5
    #11.08, tirando o limiar plt.axis([0,epochs_grid,0,limiar]), tentando mudar para escala logaritmica
    # plt.axis([0,epochs_grid,-0.6,0.2])
    # plt.xlabel('Epoch')
    # plt.ylabel('Log-Loss')
    # plt.grid()
    # plt.savefig(path+'Loss_plot-All_Folds.png')
    # plt.clf()
    plt.axis([0,epochs_grid,0,limiar])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(path+'Loss_plot-All_Folds.png')
    plt.clf()

# Função que plota gráfico de predição de um fold  
# def graphic_of_fold_predictions(df_pressures,df_prediction,fold_index,path,avg_correlation, val_loss):
#     df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
#     df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
#     plt.title('Pred_'+str(fold_index)+'; MSE: '+ str("%.4f" % val_loss)+ '; Corr: '+ str("%.4f" % avg_correlation[0]) )
#     plt.xlabel('Time[s]')
#     plt.ylabel('Amplitude')
#     plt.savefig(path+'Predicition-Fold_'+str(fold_index)+'.png')
#     plt.clf()

# Função que plota gráfico de predição de cada video  
def graphic_of_video_predictions(df_pressures,df_prediction,fold_index,video_index,path):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    df_pressures.plot(color='orange',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    plt.title('Predicition-Fold_'+str(fold_index))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Predicition-Fold_'+str(fold_index)+'-'+video_index+'.png')
    plt.clf()

# Função que plota gráfico de predição de cada fold  
def graphic_of_fold_predictions(df_pressures,df_prediction,fold_index,path,val_loss,avg_correlation,begin):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    y = [10, 12, 14, 16]
    plt.yticks(y)

    df_pressures.plot(color='black',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='red',alpha=0.5,ax=plt.gca())
    
    
    val = round(val_loss[fold_index],6)
    correlation = round(avg_correlation[fold_index][0],6)
    
    plt.title('Pred_'+str(fold_index)+';'+str(begin)+'; MSE: '+ str(val)+ '; Corr: '+ str(correlation) )
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Predicition-Fold_'+str(fold_index)+'.png')
    plt.clf()

def graphic_of_fold_train_predictions(df_pressures,df_prediction,fold_index,path,val_loss,avg_correlation,begin):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    y = [10, 12, 14, 16]
    plt.yticks(y)

    df_pressures.plot(color='black',alpha=1.0,ax=plt.gca())
    df_prediction.plot(color='blue',alpha=0.5,ax=plt.gca())
    
    
    val = round(val_loss[fold_index],6)
    correlation = round(avg_correlation[fold_index][0],6)
    
    plt.title('Pred_'+str(fold_index)+';'+str(begin)+'; MSE: '+ str(val)+ '; Corr: '+ str(correlation) )
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Train_Predicition-Fold_'+str(fold_index)+'.png')
    plt.clf()

def graphic_of_fold_val_train_predictions(df_val_pressures,df_val_prediction,df_train_pressures,df_train_prediction,fold_index,path,begin):
    predict_path = os.path.join(path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    y = [10, 12, 14, 16]
    plt.yticks(y)


    df_val_pressures.index  = range(len(df_train_pressures), len(df_train_pressures) + len(df_val_pressures))
    df_val_prediction.index = range(len(df_train_prediction), len(df_train_prediction) + len(df_val_prediction))

    concat_dfs_pressures = pd.concat([df_train_pressures, df_val_pressures], axis=0)
    concat_dfs_predictions = pd.concat([df_train_prediction, df_val_prediction], axis=0)
    
    concat_dfs_pressures.plot(color='gray',alpha=1.0,ax=plt.gca())
    concat_dfs_predictions.plot(color=['blue','red'],alpha=0.5,ax=plt.gca())
    
    plt.title('Pred_'+str(fold_index)+';'+str(begin))
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.savefig(predict_path+'Train_val_Predicition-Fold_'+str(fold_index)+'.png')
    plt.clf()

# Função que salva o estado dos códigos no início do treino
def save_current_version_of_codes(time_file, LSTM):
    if LSTM:
        source_train = os.getcwd() + "/train.py"
        destination_train = os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/train.py"
        
        source_utils = os.getcwd()+"/utils.py"
        destination_utils = os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/utils.py"

        source_functions = os.getcwd()+"/functions.py"
        destination_functions = os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/functions.py"

        os.system('cp '+source_train+' '+destination_train)
        os.system('cp '+source_utils+' '+destination_utils)
        os.system('cp '+source_functions+' '+destination_functions)

        os.system('mv '+destination_train+' '+os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/train-"+time_file+".py")
        os.system('mv '+destination_utils+' '+os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/utils-"+time_file+".py")
        os.system('mv '+destination_functions+' '+os.getcwd()+"/res_tcc/"+"LSTM_"+time_file+"/functions-"+time_file+".py")
    else:
        source_train = os.getcwd() + "/train.py"
        destination_train = os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/train.py"
        
        source_utils = os.getcwd()+"/utils.py"
        destination_utils = os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/utils.py"

        source_functions = os.getcwd()+"/functions.py"
        destination_functions = os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/functions.py"

        os.system('cp '+source_train+' '+destination_train)
        os.system('cp '+source_utils+' '+destination_utils)
        os.system('cp '+source_functions+' '+destination_functions)

        os.system('mv '+destination_train+' '+os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/train-"+time_file+".py")
        os.system('mv '+destination_utils+' '+os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/utils-"+time_file+".py")
        os.system('mv '+destination_functions+' '+os.getcwd()+"/res_tcc/"+"VGG_"+time_file+"/functions-"+time_file+".py")


def graphic_of_training_bt(FOLD,train_lstm,train_mlp,val_lstm,val_mlp, path):
    print(train_lstm)
    plt.plot(train_lstm, label="LSTM train")
    plt.plot(train_mlp, label="MLP train")
    plt.plot(val_lstm, label="LSTM val")
    plt.plot(val_mlp, label="MLP val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(path+'LSTM X MLP'+str(FOLD)+'.png')
    plt.clf()
