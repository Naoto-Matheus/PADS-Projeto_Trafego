#Listas necessárias para a busca por parametros
epochs = [100]
opt = ['sgd']
batch = [32]
dropout = [0.5] 
dropout_lstm = [0.2]
lr = [1e-04]

SCHEDULER = False #LR Scheduler
NUMBER_STEPS_EPOCHS = 30
lr_scheduler = [1e-04,5e-05,1e-05]


name_Folds = '10F' #10F , 308, 344, LM, -> para mudar o dataset de teste mudar o arquivo folds_teste
#f4 é fold 4 presente no 10F
folds_number = 10 #Se for treino a quantidade em name_Fols, se for teste, a quantidade de Folds do treino


# Constantes 
LSTM =  True
FEATURES = 'Felipe'
#'MatheusNaotoF' 'MatheusNoPIL''MatheusNaoto' ou 'test_features' ou 'Felipe' ou 'Matheus' ou 'torch_model_with_weights_of_tf/keras' 
size_windows = 32 
num_layers = 1 # Número de camadas lstm empilhadas no modelo
bidirectional = False
option_overlap = True
option_causal = False


OPTION_SHUFFLE = True #False para test(?)
OPTION_NUM_WORKERS = 4
INPUT_SIZE_FEATURES = 512
HIDDEN_SIZE = 32
OUTPUT_SIZE_FEATURES = 1
SEED_NUMBER = 22

PATH_DATA_TO_EXTRACTION = '/home/caroline/traffic-analysis/dataset/' # Caminho dos dados usados para a extração
PATH_EXTRACTED_FEATURES = '/home/caroline/Traffic-Pytorch/Data/Preprocessed/' # Caminho onde salvo as features e os targets  '
EXTRACTION_MODEL = 'vgg16'
PATH_FEATURES_CAROL = PATH_EXTRACTED_FEATURES+EXTRACTION_MODEL+'/'
VIDEOS_NUMBER = 38
PATH_FEATURES_FELIPE = '/home/felipevr/traffic_sound/dataset/preprocessed/features/vgg16/'
#PATH_FEATURES_FELIPE = '/home/matheusnaoto/Debug/dataset/preprocessed/features/vgg16_3/'
PATH_FEATURES_MATHEUSN = '/home/matheusnaoto/Debug/dataset/preprocessed/features/vgg16_5/'
PATH_FEATURES_MATHEUSNOPIL = '/home/matheusnaoto/traffic_sound/dataset/preprocessed/features/Matheus_Naoto/noPIL/'
PATH_FEATURES_MATHEUSNAOTOF = '/home/matheusnaoto/dataset/preprocessed/features/matheusNaoto/'
PATH_TARGETS_FELIPE = '/home/felipevr/traffic_sound/dataset/preprocessed/targets/'
PATH_FEATURES_TF_KERAS = '/home/caroline/traffic_sound/src/extraction/traffic_sound/dataset/preprocessed/features/'
PATH_FEATURES_TEST_VIDEOS ='/home/matheusnaoto/traffic_sound/dataset/preprocessed/features/using/'
PATH_NMB_FEATURES = '/home/matheusnaoto/Traffic-videos/dataset/'

CONST_STR_DATASET_FOLDS_DATAPATH = "/home/mathlima/dataset/folds/"
CONST_STR_DATASET_VIDEO_DATAPATH = "/home/mathlima/dataset/"


NOF_308_344 = "/home/matheusnaoto/source/treino/nof308344/"
# Lista de vídeos
videos_list = [
    "M2U00001MPG",
    "M2U00002MPG",
    "M2U00003MPG",
    "M2U00004MPG",
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00007MPG",
    "M2U00008MPG",
    "M2U00012MPG",
    "M2U00014MPG",
    "M2U00015MPG",
    "M2U00016MPG",
    "M2U00017MPG",
    "M2U00018MPG",
    "M2U00019MPG",
    "M2U00022MPG",
    "M2U00023MPG",
    "M2U00024MPG",
    "M2U00025MPG",
    "M2U00026MPG",
    "M2U00027MPG",
    "M2U00029MPG",
    "M2U00030MPG",
    "M2U00031MPG",
    "M2U00032MPG",
    "M2U00033MPG",
    "M2U00035MPG",
    "M2U00036MPG",
    "M2U00037MPG",
    "M2U00039MPG",
    "M2U00041MPG",
    "M2U00042MPG",
    "M2U00043MPG",
    "M2U00045MPG",
    "M2U00046MPG",
    "M2U00047MPG",
    "M2U00048MPG",
    "M2U00050MPG",
    "M2U00073MPG",
    "M2U00075MPG",
    "M2U00081MPG",
    "M2U00084MPG",
    "M2U00093MPG",
    "M2U00095MPG",
    "M2U00096MPG",
    "M2U00101MPG",
    "M2U00088MPG",
    "M2U00104MPG"
]

# ---------------- PRINT FUNCTIONS ------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_train(s):
    print(f"{bcolors.OKBLUE}[TRAIN]: {bcolors.ENDC}" + s)
def print_info(s):
    print(f"{bcolors.OKGREEN}[INFO]: {bcolors.ENDC}"+s)
def print_error(s):
    print(f"{bcolors.FAIL}[ERROR]: {bcolors.ENDC}"+s)
def print_debug(s):
    print(f"{bcolors.HEADER}[DEBUG]: {bcolors.ENDC}"+s)
def print_warning(s):
    print(f"{bcolors.WARNING}[DEBUG]: {bcolors.ENDC}"+s)

