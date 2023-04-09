# https://note.yu9824.com/howto/2022/05/07/conda-package-release.html
# https://zenn.dev/akikuno/articles/c216cae0e4e0f6

# オフィシャルガイド
# https://conda-forge.org/docs/maintainer/00_intro.html
conda activate nqlib
conda install conda-build
conda install -c conda-forge grayskull

cd H:\マイドライブ\IshikawaMinamiLab\研究\NQLib\conda-release
# conda skeleton pypi nqlib
grayskull pypi nqlib  # 前回のマージでこれを使うべきだと言われた．

git clone https://github.com/knttnk/staged-recipes.git

# nqlib という名前のブランチ名を作成してそこに移動
git checkout -b nqlib  cd H:\マイドライブ\IshikawaMinamiLab\研究\NQLib\conda-release\staged-recipes

# conda-release/nqlib/meta.yaml を staged-recipes/nqlib/recipes に
# meta.yaml の license_file を LICENSE.txt に変え，nqlibのそれをmeta.yamlの隣にコピー
# 移動したあと下を実行
git add .
git commit -m "restored the example and added LICENSE.txt"
git push origin nqlib  # originの後ろは、新しく作成したブランチ名

