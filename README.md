# PADS-Projeto_Trafego
## Extração de Frames e Vídeos:

Passos:

1. Colocar os vídeos em uma pasta, por convenção dentro de uma pasta raw.
    
    Exemplo: /home/matheusnaoto/AumentoBase/Raw

2. Alterar as constantes de pastas dentro do arquivo gloabals_and_functions

3. Rodar o arquivo dataset_process.py

    Nesse código o áudio será extraido do vídeo, os frames do vídeo serão extraídos com uma decimação de 10 vezes

Da forma que eu coloquei os caminhos no globals_and_functions, será criado uma pasta chamada dataset, dentro de AumentoBase e uma pasta dentro de dataset com o nome do vídeo 


├── AumentoBase/
│   ├── Raw/
│   │   └── cam1_2025-06-08_19-26-31.mp4
│   ├── dataset/
│   │    └── cam1_2025-06-08_19-26-31mp4
.   │            └── frames.png
.   │            └── output_targets.npy
.   │            └── number_of_frames.pickle
    │            └── video_statistics.pickle
    │            └── stacked_video_frames.npy
    │
    └── configuration_file.ṕickle
