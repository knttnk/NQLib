# conda が認識されないときは，
# "C:\Users\kenta\miniconda3\condabin\conda.bat" init powershell
# を実行する

# Python環境の準備
# conda create -n nqlib
# conda activate nqlib

# VSCode の Powershell で，
# ./create_env.ps1
# を実行
conda install python numpy scipy sympy matplotlib autopep8 mypy selenium ipykernel jupyter notebook -y
conda install nqlib -c conda-forge -y
python -m ipykernel install --name nqlib 
jupyter kernelspec list

# conda install -c conda-forge jupyterlab
Pause

# 環境削除
# jupyter kernelspec uninstall nqlib
# conda remove -n nqlib --all

