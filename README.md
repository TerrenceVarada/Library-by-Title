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

# 局限
本项目用的模型是 ChatGLM-6B。这里对ChatGLM-6B和ChatGPT的结果做个展示

### Q: 概括《不乖》这本书说了什么
## ChatGLM-6B
《不乖》是一本小说，作者是台湾作家朱天心。该小说讲述了一个关于家庭、亲情、友情和自我成长的故事，通过主人公的成长历程，探讨了现代社会中人们面临的各种情感问题和困境。

小说的主人公叫做张琳，她是一个出生在富裕家庭的孩子，但父母对她的过度呵护和关爱让她感到缺乏自由和独立，导致她产生了自我怀疑和迷茫。在成长过程中，张琳经历了许多挫折和困难，包括家庭变故、朋友背叛、学业困境等等，但她通过自己的努力和勇气，逐渐找到了自我认知和成长的方向。

小说通过张琳的故事，探讨了现代社会中人们面临的各种情感问题和困境，包括家庭矛盾、人际关系、自我认同、成长经历等等。同时，小说也呈现出了中国传统文化和价值观的影响，以及它们在现代社会中的转化和变化。

## ChatGPT
《不乖》这本书是由美国心理学家Richard Friedman所著，主要讲述了孩子不乖行为的根源以及如何应对和处理孩子的不乖行为。

书中首先指出了孩子不乖行为的原因，认为孩子的行为是由多种因素综合作用的结果，包括基因、家庭环境、社会文化等因素。接着，作者提出了一些应对孩子不乖行为的方法，如建立健康的亲子关系、采用积极的教育方法、培养孩子的自我控制力等。

此外，书中还介绍了一些常见的不乖行为，如挑食、说谎、攻击性行为等，以及如何分析和应对这些行为。作者还讲述了一些案例，通过具体的实例来说明如何应对孩子的不乖行为。

总的来说，《不乖》这本书主要是为父母和教育工作者提供了一些有益的指导和建议，帮助他们更好地理解孩子的行为，采取正确的方法来应对和处理孩子的不乖行为，从而建立积极的亲子关系，促进孩子的健康成长。


# Reference

1. [5 Levels Of Summarization - Novice To Expert](https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb)
2. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
3. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
