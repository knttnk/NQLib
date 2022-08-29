# https://note.yu9824.com/howto/2022/05/07/conda-package-release.html
conda activate base
conda install conda-build
conda install -c conda-forge grayskull

cd H:\マイドライブ\IshikawaMinamiLab\研究\NQLib\conda-release
conda skeleton pypi nqlib
# grayskull pypi nqlib  # 失敗したときはこれを試してみる

