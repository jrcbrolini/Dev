conda create -n myenv python=3.11
conda activate myenv
conda deactivate
conda remove --name myenv --all
conda env remove --name myenv


pip freeze > requirements.txt
python -m pip install -r requirements.txt