
from utils import name_Folds, folds_number


# Inicializando o dicionário para cada fold
folds = [dict() for i in range(folds_number)]

if name_Folds == '10F':
    folds[0]["train"] = [
        "M2U00004MPG",
        "M2U00006MPG",
        "M2U00008MPG",
        "M2U00012MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00018MPG",
        "M2U00022MPG",
        "M2U00024MPG",
        "M2U00026MPG",
        "M2U00027MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00035MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[0]["val"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00005MPG",
        "M2U00007MPG",
        "M2U00019MPG",
        "M2U00023MPG",
        "M2U00025MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00036MPG"
    ]
    folds[1]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00008MPG",
        "M2U00012MPG",
        "M2U00014MPG",
        "M2U00015MPG",
        "M2U00018MPG",
        "M2U00019MPG",
        "M2U00022MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00025MPG",
        "M2U00029MPG",
        "M2U00031MPG",
        "M2U00032MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[1]["val"] = [
        "M2U00004MPG",
        "M2U00007MPG",
        "M2U00017MPG",
        "M2U00016MPG",
        "M2U00026MPG",
        "M2U00027MPG",
        "M2U00030MPG",
        "M2U00033MPG",
        "M2U00035MPG"
    ]
    folds[2]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00016MPG",
        "M2U00018MPG",
        "M2U00019MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00025MPG",
        "M2U00027MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[2]["val"] = [
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00012MPG",
        "M2U00015MPG",
        "M2U00022MPG",
        "M2U00026MPG",
        "M2U00035MPG",
        "M2U00046MPG",
        "M2U00047MPG"
    ]
    folds[3]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00012MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00019MPG",
        "M2U00022MPG",
        "M2U00023MPG",
        "M2U00025MPG",
        "M2U00026MPG",
        "M2U00027MPG",
        "M2U00030MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00035MPG",
        "M2U00036MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[3]["val"] = [
        "M2U00014MPG",
        "M2U00018MPG",
        "M2U00024MPG",
        "M2U00029MPG",
        "M2U00031MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG"
    ]
    folds[4]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00019MPG",
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
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG"
    ]
    folds[4]["val"] = [
        "M2U00003MPG",
        "M2U00008MPG",
        "M2U00012MPG",
        "M2U00018MPG",
        "M2U00022MPG",
        "M2U00036MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[5]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00004MPG",
        "M2U00012MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00022MPG",
        "M2U00025MPG",
        "M2U00026MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00035MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[5]["val"] = [
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00018MPG",
        "M2U00019MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00027MPG"
    ]
    folds[6]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00016MPG",
        "M2U00018MPG",
        "M2U00019MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00025MPG",
        "M2U00027MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00035MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[6]["val"] = [
        "M2U00012MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00022MPG",
        "M2U00026MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG"
    ]
    folds[7]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00016MPG",
        "M2U00018MPG",
        "M2U00022MPG",
        "M2U00023MPG",
        "M2U00025MPG",
        "M2U00026MPG",
        "M2U00027MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00033MPG",
        "M2U00035MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00050MPG"
    ]
    folds[7]["val"] = [
        "M2U00003MPG",
        "M2U00012MPG",
        "M2U00015MPG",
        "M2U00019MPG",
        "M2U00024MPG",
        "M2U00029MPG",
        "M2U00032MPG",
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG"
    ]
    folds[8]["train"] = [
        "M2U00003MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00012MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00019MPG",
        "M2U00022MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00026MPG",
        "M2U00027MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00032MPG",
        "M2U00033MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00042MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00047MPG",
        "M2U00050MPG"
    ]
    folds[8]["val"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00004MPG",
        "M2U00008MPG",
        "M2U00018MPG",
        "M2U00025MPG",
        "M2U00035MPG",
        "M2U00039MPG",
        "M2U00041MPG",
        "M2U00046MPG",
        "M2U00048MPG"
    ]
    folds[9]["train"] = [
        "M2U00001MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00008MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00018MPG",
        "M2U00022MPG",
        "M2U00023MPG",
        "M2U00024MPG",
        "M2U00025MPG",
        "M2U00026MPG",
        "M2U00029MPG",
        "M2U00030MPG",
        "M2U00031MPG",
        "M2U00032MPG",
        "M2U00035MPG",
        "M2U00036MPG",
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00041MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[9]["val"] = [
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00007MPG",
        "M2U00012MPG",
        "M2U00016MPG",
        "M2U00019MPG",
        "M2U00027MPG",
        "M2U00033MPG",
        "M2U00042MPG",
        "M2U00047MPG"
    ]
elif name_Folds == '308':
#30/8

    folds[0]["train"] = [
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00008MPG",
        "M2U00012MPG",
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
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[0]["val"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00004MPG",
        "M2U00014MPG",
        "M2U00042MPG",
        "M2U00046MPG",
        "M2U00047MPG"
    ]
elif name_Folds == '344':
#34/4:
    folds[0]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00004MPG",
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
        "M2U00046MPG",
        "M2U00047MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]
    folds[0]["val"] = [
    "M2U00005MPG",
    "M2U00006MPG",
    "M2U00043MPG",
    "M2U00045MPG"
    ]
elif name_Folds == "f4":
    folds[0]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00004MPG",
        "M2U00005MPG",
        "M2U00006MPG",
        "M2U00007MPG",
        "M2U00014MPG",
        "M2U00017MPG",
        "M2U00015MPG",
        "M2U00016MPG",
        "M2U00019MPG",
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
        "M2U00037MPG",
        "M2U00039MPG",
        "M2U00043MPG",
        "M2U00045MPG",
        "M2U00046MPG",
        "M2U00047MPG"
    ]
    folds[0]["val"] = [
        "M2U00003MPG",
        "M2U00008MPG",
        "M2U00012MPG",
        "M2U00018MPG",
        "M2U00022MPG",
        "M2U00036MPG",
        "M2U00048MPG",
        "M2U00050MPG"
    ]


elif name_Folds == 'LM':
# L. Mazza:
    folds[0]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00005MPG",
    ]
    folds[0]["val"] = [
        "M2U00003MPG",
        "M2U00006MPG",
    ]
    folds[1]["train"] = [
        "M2U00001MPG",
        "M2U00003MPG",
        "M2U00005MPG",

    ]
    folds[1]["val"] = [
        "M2U00002MPG",
        "M2U00006MPG",
    ]
    folds[2]["train"] = [
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00005MPG",
    ]
    folds[2]["val"] = [
        "M2U00001MPG",
        "M2U00006MPG",
    ]
    folds[3]["train"] = [
        "M2U00001MPG",
        "M2U00002MPG",
        "M2U00006MPG",
    ]
    folds[3]["val"] = [
        "M2U00003MPG",
        "M2U00005MPG",
    ]
    folds[4]["train"] = [
        "M2U00001MPG",
        "M2U00003MPG",
        "M2U00006MPG",
    ]
    folds[4]["val"] = [
        "M2U00002MPG",
        "M2U00005MPG",
    ]
    folds[5]["train"] = [
        "M2U00002MPG",
        "M2U00003MPG",
        "M2U00006MPG",
    ]
    folds[5]["val"] = [
        "M2U00001MPG",
        "M2U00005MPG",
    ]
    folds[6]["train"] = [
        "M2U00001MPG",
        "M2U00006MPG",
        "M2U00003MPG",
    ]
    folds[6]["val"] = [
        "M2U00008MPG",
    ]
#Folds de Teste
elif name_Folds == 'teste2':
    folds[0]["test"] = [
        "M2U00004MPG",
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
elif name_Folds == 'teste1':
    folds[0]["test"] = [
    "M2U00073MPG",
    "M2U00075MPG",
    "M2U00081MPG",
    "M2U00084MPG",
    "M2U00093MPG",
    "M2U00095MPG",
    "M2U00096MPG",
    "M2U00101MPG"
]
elif name_Folds == 'teste_todos':
    folds[0]["test"]=[
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


# folds[0]["test"] = [
#     "M2U00073MPG",
#     "M2U00075MPG",
#     "M2U00084MPG",
#     "M2U00096MPG",
# ]



