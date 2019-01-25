# ReadMe

另：本人搜集了个人笔记并整理成册，命名为《AI算法工程师手册》，详见:www.huaxiaozhuan.com

## 1. 源码结构

这里给出主要的目录结构。其中 `sphinx` 自动生成的目录和文件未全部列出。

```
book/
		docs/ 	.......................> 说明文档
				make.bat ...............> sphinx 脚本
				build/...................> sphinx 生成的文档所在目录
						html/............> sphinx 生成的 HTML文档的目录
				source/..................> sphinx 的配置文件以及生成的 .rst 文件
						conf.py..........> sphinx 的配置文件
		chapters/ ........................> 源代码
				Bayesian/...................> 朴素贝叶斯和贝叶斯网络	
				Cluster_EM/.................> 聚类和 EM 算法
				Decision_Tree/..............> 决策树
			 	Ensemble/...................> 集成学习
				KNN_Dimension_Reduction/....> KNN和降维
				Linear/.....................> 线性模型
				Model_Selection/............> 模型选择
				Perceptron_Neural_Network/..> 感知机和神经网络
				PreProcessing/..............> 数据预处理
				Semi_Supervised_Learning....> 半监督学习
				SVM/........................> 支持向量机
				Kaggle/.....................> Kaggle 实战
```

## 2. 使用 sphinx 


使用 `sphinx`自动生成文档主要利用了 `sphix`的 `autodoc` 功能。这里的 `conf.py` 已经配置好。生成文档需要两步：

1. 进入命令行后，切换到 `book/`文件夹下
2. 在命令行中输入命令：

	```
	sphinx-apidoc -o docs/source chapters
	```
	该命令将会从 `chapters`目录下的`.py`文件中的抽取注释生成`.rst`文档（这些文档将被存放在 `docs/source/`目录下）

3. 在命令行中输入命令：

	```
	cd docs
	make html
	```
	其中第一行命令是进入`docs/`目录。第二行命令是根据`.rst`文档生成 `html`文档（这些`html`文档位于`docs/build/html/`目录下

## 3. 修改主题

你可以修改生成的`HTML`文件的样式，这是通过修改`sphinx`的主题来实现的。

修改 `conf.py`的 `html_theme = 'classic'` 就能实现修改主题。这里我采用经典主题`'classic'`。内建的主题有：

```
'alabaster'、'sphinx_rtd_theme'、'classic'、'sphinxdoc'、'scrolls'、'agogo'、
'traditional'、 'nature'、 'haiku'、'pyramid bizstyle'

```

## 4. 源码注释

源码注释的格式为：

```
def func(a,b):
    '''
	函数的描述
    
    :param a:  参数 a 的描述
    :param b: 参数 b 的描述 
    :return:  返回值的描述
    '''
    pass
```

这里要注意空行的空格的存在。如果没有这些空格和空行，则 `sphinx`可能会误判这些注释的意义。
