# Library-by-Title
# 项目概述：
这个项目做了一件很简单的事：
1. 根据文件名获得书籍名称；
2. 构造 prompt 输入 LLM 生成书籍摘要；
3. 将摘要生成 Embedding；
4. 聚类 Embedding；
# 安装说明
项目运行在 Mac M1 上，python版本是 3.10.11

pip install -r requirements.txt
# 使用说明
配置 config.py 运行 run.py即可在目标文件夹获得新的书籍分类。

run.py 需要配置的参数是 cluster_search（是否搜索聚类参数）和 max_clusters（最大聚类数）

1. get_title_path 产出书籍名称及文件路径；
2. get_embedding 产出书名概要和概要的embedding；
3. get_cluster 产出书籍归类明细及每个cluster的中心点 index；
4. move_files 将书籍按照 cluster 的结果进行移动；

# Reference

1. [5 Levels Of Summarization - Novice To Expert](https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb)
2. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
3. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
