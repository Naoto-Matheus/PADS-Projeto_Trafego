from distutils.log import error
import numpy as np
import os
import pandas as pd
import itertools
from itertools import product
import random
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import *
from Myfolds import *
from functions_refact import *





def writeOnLog(FILE_PATH, text):
    with open(FILE_PATH, 'a') as history_file:
        history_file.write(text + '\n')


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    TIME_FILE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    FILE_NAME = 'Training_report-'+ TIME_FILE  
if LSTM:
    hyperparams_list = [epochs, opt, batch, dropout, lr, dropout_lstm]
else:
    hyperparams_list = [epochs, opt, batch, dropout, lr]

permutations = list(itertools.product(*hyperparams_list))

# Listas para armazenar resultados
train_list = pd.DataFrame()
val_list = pd.DataFrame()
sum_val_loss = 0
sum_val_correlation = 0
best_vals_loss = []
best_train_loss = []
best_vals_epochs = []
best_vals_correlation = []

# here i can use if i want to make a grid search
for permutation_index, perm in enumerate(permutations):
    if LSTM:
        epochs_grid, optimizer_grid, batch_grid, dropout_grid, lr_grid, dropout_lstm_grid = perm
    else:
        epochs_grid, optimizer_grid, batch_grid, dropout_grid, lr_grid = perm


    TRAINING_RESULTS_PATH = os.path.join(
        "res_tcc/",
        ('LSTM_' if LSTM else 'VGG_') + TIME_FILE + "/Grid_" + str(permutation_index) + "/"
    )
    print_info(TRAINING_RESULTS_PATH)



    if not os.path.exists(TRAINING_RESULTS_PATH):
        os.makedirs(TRAINING_RESULTS_PATH)
    
    save_current_version_of_codes(TIME_FILE, LSTM)      
    
    FILE_PATH = TRAINING_RESULTS_PATH+FILE_NAME+"-Grid_"+str(permutation_index)+'.txt'
    
    #DOING THIS JUST TO WRITE IN THE LOG
    if LSTM:
        hyperparams_dict = {
            'epochs': perm[0],
            'optimizer': perm[1],
            'batch': perm[2],
            'dropout': perm[3],
            'lr': perm[4],
            'dropout_lstm': perm[5]
        }
    else:
        hyperparams_dict = {
            'epochs': perm[0],
            'optimizer': perm[1],
            'batch': perm[2],
            'dropout': perm[3],
            'lr': perm[4]
}
    
    writeOnLog(FILE_PATH,'\nDispositivo usado pelo Pytorch: {}\n'.format(device))
    writeOnLog(FILE_PATH,'\nFeatures usadas: {}\n'.format(FEATURES))
    writeOnLog(FILE_PATH,'\n*****************************************\n')
    writeOnLog(FILE_PATH,'\n-----> GRID {}: {}\n'.format(permutation_index,hyperparams_dict))
    
        
    
    if(LSTM == True):
        writeOnLog(FILE_PATH,'\n LSTM ={}\n\n'.format(LSTM))
        writeOnLog(FILE_PATH,'\n option_overlap={} , option_causal={} ,option_dropout_lstm={}\n\n'.format(option_overlap,option_causal,dropout_lstm_grid))
        


    
    for fold_index in range(folds_number): 
        print_info(f'-> Fold {fold_index}')
        writeOnLog(FILE_PATH,'\n\n--> Fold: {}\n'.format(fold_index))

        torch.manual_seed(SEED_NUMBER)
        np.random.seed(SEED_NUMBER)
        random.seed(SEED_NUMBER)
        
        

        if(LSTM == True):
            print_info("Dataset: LSTM ")
            train_dataset = LSTM_Dataset(folds,         
                                            fold_index,     
                                            'train',        
                                            option_overlap, 
                                            option_causal,  
                                            size_windows)                                                                                                                                                                                                                                                    

        else:
            print_info("Dataset: VGG ")
            train_dataset = VGG_Dataset(folds,          
                                        fold_index,     
                                        mode='train')   
                                                                        
                                                                        
        
        train_frames_array = [] 
        train_frames_tensor = [] 
        train_pressures_array = []
        val_frames_array = []
        val_pressures_array = []

        
        if(LSTM == True):
            val_dataset = LSTM_Dataset( folds,         
                                        fold_index,     
                                        'val',          
                                        option_overlap, 
                                        option_causal,  
                                        size_windows)                                                                                                                                                                                                                                                     
        else:
            val_dataset = VGG_Dataset(folds,        
                                        fold_index,   
                                        mode='val')   
                

        len_val = len(val_dataset)
        
        #Val_pressures_array is needed to make the correlation 
        for index in range(len_val):
            val_frame,val_pressure,_ = val_dataset[index] 

            if(LSTM == True) and (option_causal == False):
                val_frames_array.append(val_frame[size_windows//2 - 1])

            elif(LSTM == True) and (option_causal == True):
                val_frames_array.append(val_frame[size_windows - 1])
            
            else:
                val_frames_array.append(val_frame)
            
            val_pressures_array.append(val_pressure)
            
        
        writeOnLog(FILE_PATH,'\nFormat of train data: frames: {} , pressures: {} \n'.format(len(train_frames_array),len(train_pressures_array)))
        writeOnLog(FILE_PATH,'\nFormat of validation data: frames: {} , pressures: {} \n'.format(len(val_frames_array),len(val_pressures_array)))
        writeOnLog(FILE_PATH,'\nOption shuffle: {}\n'.format(OPTION_SHUFFLE))
        

        
        if(LSTM == True):
            modelo = LSTM_Network(INPUT_SIZE_FEATURES,OUTPUT_SIZE_FEATURES,HIDDEN_SIZE,dropout_grid,num_layers,dropout_lstm_grid,bidirectional,use_batchnorm=False) 
        else:
            modelo = VGG_Network(INPUT_SIZE_FEATURES,OUTPUT_SIZE_FEATURES,[128],dropout_grid)
        
        model = modelo.to(device)
        
        loss_function = nn.MSELoss()

        optimizer = optimizer_config(perm[1],model,lr_grid)
        
        
        list_loss_train = []
        list_loss_val = []
        list_predictions = []
        list_of_all_predictions = []
        list_of_all_train_predictions = []

        begin = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        writeOnLog(FILE_PATH,'\nBegin of training: {}'.format(begin))
        writeOnLog(FILE_PATH,'\n\nEpoch,train_loss,val_loss,lr')
        

        min_val_loss = np.inf

        for epochs_index in range(epochs_grid): 
            
            if (SCHEDULER):
                if (epochs_index < NUMBER_STEPS_EPOCHS):
                    optimizer = optimizer_config(optimizer_grid,model,lr_scheduler[0])
                    lr_grid = lr_scheduler[0]

                elif (epochs_index >= NUMBER_STEPS_EPOCHS and epochs_index <= (2*NUMBER_STEPS_EPOCHS)-1):
                    optimizer = optimizer_config(optimizer_grid,model,lr_scheduler[1])
                    lr_grid = lr_scheduler[1]

                elif (epochs_index > (2*NUMBER_STEPS_EPOCHS)-1  ):
                    optimizer = optimizer_config(optimizer_grid,model,lr_scheduler[2])
                    lr_grid = lr_scheduler[2]

            else:
                
                optimizer = optimizer_config(optimizer_grid,model,lr_grid)     
            

            train_loss = train(model,train_dataset,loss_function,optimizer,batch_grid)

            val_loss, min_val_loss, improved = validation(model,val_dataset,loss_function,TRAINING_RESULTS_PATH,fold_index,batch_grid, min_val_loss)
                
            
            print_debug('VAL_LOSSS: '+str(val_loss)+' / MIN_VAL_LOSS'+ str(min_val_loss) )
            #on the 1st epoch min_val = val_loss
                            
            if improved:
                epoch_of_min_val_loss = epochs_index + 1
            print_debug(' / BEST EPOCH: '+ str(epoch_of_min_val_loss))
            

            writeOnLog(FILE_PATH,'\n{},{},{},{}'.format(epochs_index+1,train_loss,val_loss,lr_grid))
            

            list_loss_train.append(train_loss)
            list_loss_val.append(val_loss)

            print_train(f'Epoch: {epochs_index+1}\t Training Loss: {train_loss} \t Val Loss: {val_loss}')

        end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        
        df_train = pd.DataFrame(list_loss_train, columns = ['train_loss'])
        train_loss_min = df_train['train_loss'].values.min()
        df_val = pd.DataFrame(list_loss_val, columns = ['val_loss'])
        val_loss_min = df_val['val_loss'].values.min()
        sum_val_loss += val_loss_min
        best_vals_loss.append(val_loss_min)
        best_train_loss.append(train_loss_min)
        best_vals_epochs.append(epoch_of_min_val_loss)
        
        
        #creating files training (loss x epoch ) graphs
        training_files_path = os.path.join(TRAINING_RESULTS_PATH,'files_training/')
            
        if not os.path.exists(training_files_path):
            os.makedirs(training_files_path)

        df_train.to_csv(training_files_path+"train_loss_fold_"+str(fold_index)+".csv", columns = ['train_loss'])
        df_val.to_csv(training_files_path+"val_loss_fold_"+str(fold_index)+".csv", columns = ['val_loss'])

        graphic_of_training(df_train,df_val,fold_index,training_files_path,epochs_grid)
    
        writeOnLog(FILE_PATH,'\nEnd of training: {}\n'.format(end))
        writeOnLog(FILE_PATH,'\nLoss min training: {}\n'.format(train_loss_min))
        writeOnLog(FILE_PATH,'\nLoss min val: {}\n'.format(val_loss_min))
        writeOnLog(FILE_PATH,'\nBest epoch: {}\n'.format(epoch_of_min_val_loss))
            
            # HERE THE TRAINING ENDS FROM NOW ON ITS JUST THE GRAPHS
    
        try: 
            
            best_epoch_model_path = os.path.join(TRAINING_RESULTS_PATH,'best_model/'+'best_model_fold_'+str(fold_index)+'.pth')
            model.load_state_dict(torch.load(best_epoch_model_path))
            model.to(device)
            model.eval()
                
        except: 
            print_error('Erro em abrir arquivo!')
        else:
            list_predictions = []
            pred_loader = DataLoader(val_dataset,batch_size=batch_grid,shuffle=False,num_workers=OPTION_NUM_WORKERS)
            with torch.no_grad():
                for frames,pressure in pred_loader:        
                    frames= frames.to(device)
                    pred = model(frames)
                    list_predictions.append(pred)
                
        for batch_index in list_predictions:
            for prediction_index in range(batch_index.shape[0]):
                list_of_all_predictions.append(batch_index[prediction_index])
        print_info(f'Length of All Predictions List: {len(list_of_all_predictions)}')
        
        list_of_all_predictions = np.array(list_of_all_predictions,dtype= 'float64')
        list_of_all_predictions = np.squeeze(list_of_all_predictions)
        
        val_pressures_array = np.array(val_pressures_array,dtype= 'float64')
        val_pressures_array = np.squeeze(val_pressures_array)

        
        correlation = pearsonr(val_pressures_array ,list_of_all_predictions)
        
        
        sum_val_correlation += correlation[0]
        best_vals_correlation.append(correlation)
        
        writeOnLog(FILE_PATH,'\nCorrelation: {}\n\n\n'.format(correlation))
        

        
        avg_val_loss = sum_val_loss/folds_number
        avg_correlation = sum_val_correlation/folds_number
        
        predict_files_path = os.path.join(TRAINING_RESULTS_PATH,'files_predictions/')
            
        if not os.path.exists(predict_files_path):
            os.makedirs(predict_files_path)

        np.save(predict_files_path +'predictions_fold_'+str(fold_index),list_of_all_predictions)

        df_pressure = pd.DataFrame(val_pressures_array,columns=['Real_samples_best_model'])
        df_prediction = pd.DataFrame(list_of_all_predictions,columns=['Val_Predicted_samples_best_model'])
                
        graphic_of_fold_predictions(df_pressure,df_prediction,fold_index,predict_files_path, best_vals_loss, best_vals_correlation, TIME_FILE)

        
        begin_index = 0
        auxiliary = 0
        # Doing the predictions graphs
        for video_index in folds[fold_index]['val']:

            

            if (FEATURES == 'Felipe'):
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE,video_index+'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)
            if (FEATURES == 'MatheusNoPIL'):
                training_targets = np.load(os.path.join(PATH_TARGETS_FELIPE,video_index+'_targets.npy'))
                training_targets = np.mean(training_targets, axis=1)
                
            if (FEATURES == 'Matheus'):
                
                path_matheus = '/home/mathlima/dataset/folds/vgg16/'
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)
            
            if (FEATURES == 'MatheusNaoto'):
                training_targets = np.load('/home/mathlima/dataset/' + video_index +'/output_targets.npy')
                training_targets = np.mean(training_targets, axis=1)  

            if (FEATURES == 'MatheusNaotoF'):
            
                
                if (name_Folds == 'f4'):
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_4_test_output_data.npy')
                else:
                    training_targets = np.load('/home/mathlima/dataset/folds/' +'fold_'+str(fold_index)+'_test_output_data.npy') 


            number_frames =  training_targets.shape[0] 
            auxiliary = auxiliary + number_frames

            df_prediction['Val_Predicted_samples_best_model'][begin_index:auxiliary]

            predict_files_path_vgg = os.path.join(predict_files_path,'predictions_of_each_video_in_fold_'+str(fold_index)+'/')
            
            if not os.path.exists(predict_files_path_vgg):
                os.makedirs(predict_files_path_vgg)
        
            graphic_of_video_predictions(df_pressure['Real_samples_best_model'][begin_index:auxiliary],
                                        df_prediction['Val_Predicted_samples_best_model'][begin_index:auxiliary],
                                        fold_index,
                                        video_index,
                                        predict_files_path_vgg)
            begin_index = auxiliary


                
        

        
        val_list.loc[:,'val_loss_'+str(fold_index)]=df_val['val_loss']
        train_list.loc[:,'train_loss_'+str(fold_index)]=df_train['train_loss']
        
    
    graphic_of_training_all_folds(train_list,val_list,training_files_path,epochs_grid)
    
    
    for x in range(len(best_vals_loss)):
        writeOnLog(FILE_PATH,'\n Fold {}: \n'.format(x))
        writeOnLog(FILE_PATH,'Loss min val: {}\n'.format(best_vals_loss[x]))
        writeOnLog(FILE_PATH,'Best epoch {}\n'.format(best_vals_epochs[x]))
        writeOnLog(FILE_PATH,'Correlation: {}\n'.format(best_vals_correlation[x]))
    writeOnLog(FILE_PATH,'\nAverage val loss: {}\n'.format(avg_val_loss))
    writeOnLog(FILE_PATH,'\nAverage correlation: {}\n'.format(avg_correlation))

