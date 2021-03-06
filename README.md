# 天池大数据竞赛平台-FDDC2018金融算法挑战赛01－A股上市公司季度营收预测(排名29/2724)
[大赛链接](https://tianchi.aliyun.com/competition/entrance/231660/introduction?spm=5176.12281915.0.0.730b10bdRLKLi5)
## 数据集
赛题用到的数据包括历史财务数据、宏观数据、行情数据、行业数据。各数据包含的主要字段的名词解析以及财务数据的中英文对照。说明位于`data/0-Read me_CNS_20180524.pdf`

- 财务数据包括三张表，分别为资产负债表Balance Sheet、利润表 Income Statement、现金流量表Cash Flow Statement。
其中，由于非金融上市公司、证券、银行、保险四大行业的财务报表在结构上存在差异，所以每个类别又分为4个相对应的文档（csv格式）。
这三张表代表了一个公司全部的财务信息，三大财务报表分析是投资的基础。

- **资产负债表**：代表一个公司的资产与负债及股东权益，资产负债表是所有表格的基础

- **利润表**：代表一个公司的利润来源，而净利润则直接影响资产负债表中股东权益的变化

- **现金流量表**：代表一个公司的现金流量，更代表资产负债表的变化。现金流量表是对资产负债表变化的解释。
现金的变化最终反映到资产负债表的现金及等价物一项。而现金的变化源泉则是净利润。净利润经过“经营”、“投资”、“筹资”三项重要的现金变动转变为最终的现金变化。

- **宏观数据 Macro Industry**：一系列宏观经济学的统计指标， 包括生产总值(GDP)、国民总收入（GNI）、劳动者报酬、消费水平等。宏观经济周期是影响周期性行业的关键因素之一，对上市公司的经营情况也有直接的影响。

- **行业数据 Industry Data 行业数据**：可以指示某个行业的发展态势，上市公司都会有自己所在的行业，分析行业的发展趋势、所处阶段等可对上市公司经营情况做出大体的判断（如从汽车行业每月的销量数据中，可以看到行业的景气程度）。

- **公司经营数据 Company Operation Data**：一般为月度数据，代表特定公司主营业务月度的统计值，与公司营收密切相关，每个公司指标不一样。

- **行情数据 Market Data 行情数据**：代表上市公司股票月度交易行情，主要包括价格、成交量、成交额、换手率等。


## 特征工程

- 对每家公司关联三表及宏观数据该年的指标数据和行业数据的行业特征，取特征的滞后1期、2期、3期和1年、2年、3年的特征。
代码位于`codes/data_transform.py`

- 考虑到公司的经营跟所处的行业之间存在着经济意义的显著关系，构建同行业的所有公司的财务滞后1期、2期、3期和1年、2年、3年的特征。
代码位于`codes/data_transform.py`

- 按照行业分类，对不同的行业挑选不同的财务特征


## 模型训练
- 使用向量自回归模型，用模型中所有当期变量对不同行业的重要财务指标的若干滞后变量进行回 归，用来估计联合内生变量的动态关系(AIC、BIC 选择阶数)

- arima预测
  - 对营业收入直接建立 arima 预测
  
  - **营业收入 = 主营业务收入 + 其他业务收入**，对主营业务收入、其他业务收入序列建立 arima 预测之后再加和得到营业收入

- Prophet模型

- Xgboost，小样本Xgboost效果优于Lightgbm，关联同行业的所有公司数据，考虑时间序列性下交叉验证建立 xgboost
