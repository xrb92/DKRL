# DKRL

New: Add entity type information 


# INTRODUCTION

Description-Embodied Knowledge Representation Learning (DKRL)

Representation Learning of Knowledge Graphs with Entity Descriptions (AAAI'16)

Ruobing Xie


# COMPILE

Just type make in the folder ./


# NOTE

Pre-trained embeddings for entity/relation/word are optional.
We update both Structure-based Representations and Description-based Representations in this version. We can also fix Structure-based Representations pre-trained by other models and only update Description-based Representations.

For DKRL, we learn structure-based representations (SBR) and description-based representations (DBR) simultaneously in training.
However, Test_cnn.cpp only use description-based representations for prediction. You can load in both entity representations for joint prediction. 


# DATA

FB15k is published by the author of the paper "Translating Embeddings for Modeling Multi-relational Data (2013)." 
<a href="https://everest.hds.utc.fr/doku.php?id=en:transe">[download]</a>
You can also get FB15k from here: <a href="http://pan.baidu.com/s/1eSvyY46">[download]</a>

Entity list and descriptions of FB15k used in this work <a href="http://pan.baidu.com/s/1kUx5Wr1">[download]</a>

FB20k is based on FB15k and used for zero-shot scenario <a href="http://yun.baidu.com/s/1SAmGQ">[download]</a>

Entity type information for entity classification <a href="http://pan.baidu.com/s/1pLePRez">[download]</a>

All these datasets are also in data.rar.

Entity name file <a href="https://pan.baidu.com/s/1hsDneZE">[download]</a>



# CITE

If the codes or datasets help you, please cite the following paper:

Ruobing Xie, Zhiyuan Liu, Jia Jia, Huanbo Luan, Maosong Sun. Representation Learning of Knowledge Graphs with Entity Descriptions. The 30th AAAI Conference on Artificial Intelligence (AAAI'16).
