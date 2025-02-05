# # "Caltech101-20"

# python run.py   --dataset 0   --devices 0  --topk 0  --neg 0 --alpha 3\
#                 --beta 3 --gamma 8 --lambda1 0.1 --lambda2 0.1 --droprate 0.10\
#                 --out ./result2


# # "Scene_15"

# python run.py   --dataset 1   --devices 0  --topk 0  --neg 0 --alpha 3\
#                 --beta 3 --gamma 6 --lambda1 0.01 --lambda2 1 --droprate 0.08\
#                 --out ./result2



# # "LandUse_21"


# python run.py   --dataset 2   --devices 0  --topk 0  --neg 0 --alpha 4\
#                 --beta 2 --gamma 10 --lambda1 0.01 --lambda2 5 --droprate 0.08\
#                 --out ./result2


# # "NoisyMNIST"

# python run.py   --dataset 3   --devices 0  --topk 0  --neg 0 --alpha 4\
#                 --beta 6 --gamma 6 --lambda1 0.3 --lambda2 0.01 --droprate 0.10\
#                 --out ./result2

################################################################################


# # "Caltech101-20"

# python run.py   --dataset 0   --devices 0  --topk 0  --neg 0 --alpha 3\
#                 --beta 3 --gamma 8 --lambda1 0.1 --lambda2 0.1 --droprate 0.10\
#                 --out ./rebuttal


# # "Scene_15"

# python run.py   --dataset 1   --devices 0  --topk 0  --neg 0 --alpha 3\
#                 --beta 3 --gamma 6 --lambda1 0.01 --lambda2 1 --droprate 0.08\
#                 --out ./rebuttal



# # "LandUse_21"


# python run.py   --dataset 2   --devices 0  --topk 0  --neg 0 --alpha 4\
#                 --beta 2 --gamma 10 --lambda1 0.01 --lambda2 5 --droprate 0.08\
#                 --out ./rebuttal


# # "NoisyMNIST"

# python run.py   --dataset 3   --devices 0  --topk 0  --neg 0 --alpha 4\
#                 --beta 6 --gamma 6 --lambda1 0.3 --lambda2 0.01 --droprate 0.10\
#                 --out ./rebuttal

#####################################################################################

# "Caltech101-20"

# python run.py   --dataset 0   --devices 0  --alpha 3\
#                 --beta 3 --gamma 8 --lambda1 0.1 --lambda2 0.10 --droprate 0.10\
#                 --out ./final


# "Scene_15"

# python run.py   --dataset 1   --devices 0 --alpha 3.8\
#                 --beta 2.7 --gamma 2.2 --lambda1 0.01 --lambda2 1 --droprate 0.08\
#                 --out ./final



# "LandUse_21"


python run.py   --dataset 2   --devices 0   --alpha 3\
                --beta 3.6 --gamma 9.5 --lambda1 0.01 --lambda2 5 --droprate 0.08\
                --out ./final


# "NoisyMNIST"

# python run.py   --dataset 3   --devices 0  --alpha 3\
#                 --beta 3 --gamma 8 --lambda1 0.3 --lambda2 0.01 --droprate 0.10\
#                 --out ./final

