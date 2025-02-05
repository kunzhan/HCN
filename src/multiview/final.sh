
######################################################################################

# "Caltech101-20"


python run.py   --dataset 0   --devices 0   --alpha 2.8\
                --beta 2.9 --gamma 8.0 --lambda1 0.02 --lambda2 0.01 --droprate 0.10\
                --out ./final



# # "Scene_15"

python run.py   --dataset 1   --devices 0  --alpha 2.6\
                --beta 2.2 --gamma 2.2 --lambda1 0.01 --lambda2 1.0 --droprate 0.08\
                --out ./final







# "LandUse_21"

# python run.py   --dataset 2   --devices 0  --alpha 3.0\
#                 --beta 4.4 --gamma 8.0 --lambda1 0.01 --lambda2 5 --droprate 0.08\
#                 --out ./final




python run.py   --dataset 2   --devices 0  --alpha 3.0\
                --beta 4.4 --gamma 8.0 --lambda1 0.005 --lambda2 2.5 --droprate 0.08\
                --out ./final


###################################################################################################





