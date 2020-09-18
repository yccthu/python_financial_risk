# 萌新大赛避坑指南：天池XDatawhale零基础入门金融风控-贷款违约预测大赛

【作者】杨二茶

【声明】

1. 本指南的制作基础灵感来源于开源组织Datawhale和天池官方联合发起的0基础入门系列赛事第四场 —— 零基础入门金融风控之贷款违约预测挑战赛的第二个task。

	赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。通过这道赛题来引导大家了解金融风控中的一些业务背景，解决实际问题，帮助竞赛新人进行自我练习、自我提高。

2. 之所以是避坑指南是因为本人在实践过程中过于小白，不懂如何规避各种坑，但又对于探索多种思路和路径有很多奇【keng】思【keng】妙【wa】想【wa】，由此，有了这个指南。

3. 本人学代码时间较晚，常出现玻璃心爆表和暴怒吃键盘等不稳定状态，若本文敲代码过程中，你也出现类似症状，请跟随如下指示：

吸气【莫生气！】

呼气【莫生气！】

吸气【莫生气！】

4. 希望能帮助到和我一样毫无基础和背景的小朋友，在首次参赛的时候，少走一些弯路。

好的，做完以上的精神稳定的准备工作后，我们就可以继续第二部分**数据分析**了~

# Task2 数据分析

此部分为大赛数据准备的第二个部分，数据分析的主要目的，是为了让大家可以更清楚地了解和认识数据，抓住数据的特征，一遍进一步地完成特征工程的工作。

# 1 学习目标


1. 对竞赛的数据集进行详细地摸底，检验是否可用于或如何用于机器学习/深度学习建模，其中特别需要关注和处理的是缺失值和异常值。

2. 寻找多变量之间的相关关系，并且进行可能地因果推断和统计预测。

3. 对数据的认知提升到可进行特征工程的程度。


# 2 分析思路

## 2.1 数据概览

- 数据集整体大小、原始特征维度的了解
- info（）函数了解数据类型
- 粗略查看数据集各个特征的基本统计量

## 2.2 缺失值、唯一值、异常值

### 2.2.1 缺失值

要分析缺失值，首先，我们需要分析缺失值是如何产生的。

【产生原因】

1. 机械原因

- 测量设备损坏

2. 人为原因

- 数据难以获取
- 数据在不同时间维度采集时的样本异常
- 人为误差

【缺失值的类型】

从分布上可分为三类：

- 完全随机缺失 Missing completely at random (MCAR)
- 完全非随机缺失Missing not at random (MNAR)
- 随机缺失 Missing at random (MAR)

【产生后果】

- 对估计效率产生负面影响
- 影响预测准确率

### 2.2.2 深入数据-类型查看

- 类别型数据
- 数值型数据
    - 离散数值型数据
    - 连续数值型数据

### 2.2.3 相关关系 Correlation

- 特征与特征之间的关系
- 特征与目标变量之间的关系

### 2.2.4 pandas_profiling生成数据报告

## 2.3 代码

### 2.3.1 准备工作


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')
```


```python
t = pd.read_csv('train.csv')
```


```python
ta = pd.read_csv('testA.csv')
```

### 2.3.2 基础观察

#### 2.3.2.1 表头检查


```python
t.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>grade</th>
      <th>subGrade</th>
      <th>employmentTitle</th>
      <th>employmentLength</th>
      <th>homeOwnership</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>35000.0</td>
      <td>5</td>
      <td>19.52</td>
      <td>917.97</td>
      <td>E</td>
      <td>E2</td>
      <td>320.0</td>
      <td>2 years</td>
      <td>2</td>
      <td>...</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>18000.0</td>
      <td>5</td>
      <td>18.49</td>
      <td>461.90</td>
      <td>D</td>
      <td>D2</td>
      <td>219843.0</td>
      <td>5 years</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>12000.0</td>
      <td>5</td>
      <td>16.99</td>
      <td>298.17</td>
      <td>D</td>
      <td>D3</td>
      <td>31698.0</td>
      <td>8 years</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>11000.0</td>
      <td>3</td>
      <td>7.26</td>
      <td>340.96</td>
      <td>A</td>
      <td>A4</td>
      <td>46854.0</td>
      <td>10+ years</td>
      <td>1</td>
      <td>...</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3000.0</td>
      <td>3</td>
      <td>12.99</td>
      <td>101.07</td>
      <td>C</td>
      <td>C2</td>
      <td>54.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>...</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>




```python
ta.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>grade</th>
      <th>subGrade</th>
      <th>employmentTitle</th>
      <th>employmentLength</th>
      <th>homeOwnership</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>800000</td>
      <td>14000.0</td>
      <td>3</td>
      <td>10.99</td>
      <td>458.28</td>
      <td>B</td>
      <td>B3</td>
      <td>7027.0</td>
      <td>10+ years</td>
      <td>0</td>
      <td>...</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>6.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>800001</td>
      <td>20000.0</td>
      <td>5</td>
      <td>14.65</td>
      <td>472.14</td>
      <td>C</td>
      <td>C5</td>
      <td>60426.0</td>
      <td>10+ years</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>800002</td>
      <td>12000.0</td>
      <td>3</td>
      <td>19.99</td>
      <td>445.91</td>
      <td>D</td>
      <td>D4</td>
      <td>23547.0</td>
      <td>2 years</td>
      <td>1</td>
      <td>...</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>800003</td>
      <td>17500.0</td>
      <td>5</td>
      <td>14.31</td>
      <td>410.02</td>
      <td>C</td>
      <td>C4</td>
      <td>636.0</td>
      <td>4 years</td>
      <td>0</td>
      <td>...</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>800004</td>
      <td>35000.0</td>
      <td>3</td>
      <td>17.09</td>
      <td>1249.42</td>
      <td>D</td>
      <td>D1</td>
      <td>368446.0</td>
      <td>&lt; 1 year</td>
      <td>1</td>
      <td>...</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>



#### 2.3.2.2 dataframe形状检查


```python
t.shape
```




    (800000, 47)




```python
ta.shape
```




    (200000, 46)



通过对比两个数据集的列数，我们会发现训练集多出来了一列，因此，我们需要找出这是哪一列。

【最简思路】

用集合运算的思想，我们采用的应该是对称差集`symmetric_difference()`，对称差集可以用来查看两个集合的补集。


```python
ta.columns.symmetric_difference(t.columns)
```




    Index(['isDefault'], dtype='object')



我们会发现`isDefault`是用户“是否违约”的记录：
- 1表示存在违约情况
- 0表示没有违约情况

我们基于此来看一看样本的违约率有多少。
1. 先画图观察基本情况
2. 查看两个标签的数据情况
3. 计算违约率


```python
t['isDefault'].value_counts().plot.bar()
```




    <AxesSubplot:>




![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200918234924.png)



```python
t['isDefault'].value_counts()
```




    0    640390
    1    159610
    Name: isDefault, dtype: int64




```python
#default rate
159610/(159610+640390)
```




    0.1995125



【阶段小结1】

样本中的**用户违约率**大概在**20%**左右。

【失败思路一】
1. 先对比两列找出不同列，t.columns单独看来是一个Series，因此可以使用`map()`函数和`lambda()`函数进行组合。
2. 用t`.columns`函数来观察训练集t和测试集ta的columns
3. 观察缺少的那一列的内容


```python
t.columns.map(lambda x: 0 if x in ta.columns else 1)
```




    Int64Index([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0],
               dtype='int64')




```python
t.columns
```




    Index(['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade',
           'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
           'annualIncome', 'verificationStatus', 'issueDate', 'isDefault',
           'purpose', 'postCode', 'regionCode', 'dti', 'delinquency_2years',
           'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec',
           'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
           'initialListStatus', 'applicationType', 'earliesCreditLine', 'title',
           'policyCode', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
           'n9', 'n10', 'n11', 'n12', 'n13', 'n14'],
          dtype='object')




```python
ta.columns
```




    Index(['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'grade',
           'subGrade', 'employmentTitle', 'employmentLength', 'homeOwnership',
           'annualIncome', 'verificationStatus', 'issueDate', 'purpose',
           'postCode', 'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow',
           'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal',
           'revolUtil', 'totalAcc', 'initialListStatus', 'applicationType',
           'earliesCreditLine', 'title', 'policyCode', 'n0', 'n1', 'n2', 'n3',
           'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14'],
          dtype='object')



通过上面观察，我们知道第14列是多出的列，因此我们单独把第十四列'isDefault'提出来看，我们会发现这个列的内容


```python
t['isDefault'].value_counts()
```




    0    640390
    1    159610
    Name: isDefault, dtype: int64



【失败思路二】

当数据量较大时，直接观察很难得到我们想要的结果，因此，我们用`argwhere`函数来更为快速地找出每一列标题的中不同的。

1. 先转换为numpy的格式
2. 用`np.argwhere（）`函数来展开我们需要的列。


```python
col_tf = t.columns.map(lambda x: 0 if x in ta.columns else 1).to_list()
```


```python
col_tf = np.array(col_tf)
```


```python
np.argwhere(col_tf).flatten()
```




    array([13], dtype=int64)



我们用`t.info()`来查看第13个列，这一列是`isDefault`


```python
t.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800000 entries, 0 to 799999
    Data columns (total 47 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   id                  800000 non-null  int64  
     1   loanAmnt            800000 non-null  float64
     2   term                800000 non-null  int64  
     3   interestRate        800000 non-null  float64
     4   installment         800000 non-null  float64
     5   grade               800000 non-null  object 
     6   subGrade            800000 non-null  object 
     7   employmentTitle     799999 non-null  float64
     8   employmentLength    753201 non-null  object 
     9   homeOwnership       800000 non-null  int64  
     10  annualIncome        800000 non-null  float64
     11  verificationStatus  800000 non-null  int64  
     12  issueDate           800000 non-null  object 
     13  isDefault           800000 non-null  int64  
     14  purpose             800000 non-null  int64  
     15  postCode            799999 non-null  float64
     16  regionCode          800000 non-null  int64  
     17  dti                 799761 non-null  float64
     18  delinquency_2years  800000 non-null  float64
     19  ficoRangeLow        800000 non-null  float64
     20  ficoRangeHigh       800000 non-null  float64
     21  openAcc             800000 non-null  float64
     22  pubRec              800000 non-null  float64
     23  pubRecBankruptcies  799595 non-null  float64
     24  revolBal            800000 non-null  float64
     25  revolUtil           799469 non-null  float64
     26  totalAcc            800000 non-null  float64
     27  initialListStatus   800000 non-null  int64  
     28  applicationType     800000 non-null  int64  
     29  earliesCreditLine   800000 non-null  object 
     30  title               799999 non-null  float64
     31  policyCode          800000 non-null  float64
     32  n0                  759730 non-null  float64
     33  n1                  759730 non-null  float64
     34  n2                  759730 non-null  float64
     35  n3                  759730 non-null  float64
     36  n4                  766761 non-null  float64
     37  n5                  759730 non-null  float64
     38  n6                  759730 non-null  float64
     39  n7                  759730 non-null  float64
     40  n8                  759729 non-null  float64
     41  n9                  759730 non-null  float64
     42  n10                 766761 non-null  float64
     43  n11                 730248 non-null  float64
     44  n12                 759730 non-null  float64
     45  n13                 759730 non-null  float64
     46  n14                 759730 non-null  float64
    dtypes: float64(33), int64(9), object(5)
    memory usage: 286.9+ MB


#### 2.3.2.3 描述性统计分析

1. 观察变量类型


```python
t.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800000 entries, 0 to 799999
    Data columns (total 47 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   id                  800000 non-null  int64  
     1   loanAmnt            800000 non-null  float64
     2   term                800000 non-null  int64  
     3   interestRate        800000 non-null  float64
     4   installment         800000 non-null  float64
     5   grade               800000 non-null  object 
     6   subGrade            800000 non-null  object 
     7   employmentTitle     799999 non-null  float64
     8   employmentLength    753201 non-null  object 
     9   homeOwnership       800000 non-null  int64  
     10  annualIncome        800000 non-null  float64
     11  verificationStatus  800000 non-null  int64  
     12  issueDate           800000 non-null  object 
     13  isDefault           800000 non-null  int64  
     14  purpose             800000 non-null  int64  
     15  postCode            799999 non-null  float64
     16  regionCode          800000 non-null  int64  
     17  dti                 799761 non-null  float64
     18  delinquency_2years  800000 non-null  float64
     19  ficoRangeLow        800000 non-null  float64
     20  ficoRangeHigh       800000 non-null  float64
     21  openAcc             800000 non-null  float64
     22  pubRec              800000 non-null  float64
     23  pubRecBankruptcies  799595 non-null  float64
     24  revolBal            800000 non-null  float64
     25  revolUtil           799469 non-null  float64
     26  totalAcc            800000 non-null  float64
     27  initialListStatus   800000 non-null  int64  
     28  applicationType     800000 non-null  int64  
     29  earliesCreditLine   800000 non-null  object 
     30  title               799999 non-null  float64
     31  policyCode          800000 non-null  float64
     32  n0                  759730 non-null  float64
     33  n1                  759730 non-null  float64
     34  n2                  759730 non-null  float64
     35  n3                  759730 non-null  float64
     36  n4                  766761 non-null  float64
     37  n5                  759730 non-null  float64
     38  n6                  759730 non-null  float64
     39  n7                  759730 non-null  float64
     40  n8                  759729 non-null  float64
     41  n9                  759730 non-null  float64
     42  n10                 766761 non-null  float64
     43  n11                 730248 non-null  float64
     44  n12                 759730 non-null  float64
     45  n13                 759730 non-null  float64
     46  n14                 759730 non-null  float64
    dtypes: float64(33), int64(9), object(5)
    memory usage: 286.9+ MB



```python
ta.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Data columns (total 46 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   id                  200000 non-null  int64  
     1   loanAmnt            200000 non-null  float64
     2   term                200000 non-null  int64  
     3   interestRate        200000 non-null  float64
     4   installment         200000 non-null  float64
     5   grade               200000 non-null  object 
     6   subGrade            200000 non-null  object 
     7   employmentTitle     200000 non-null  float64
     8   employmentLength    188258 non-null  object 
     9   homeOwnership       200000 non-null  int64  
     10  annualIncome        200000 non-null  float64
     11  verificationStatus  200000 non-null  int64  
     12  issueDate           200000 non-null  object 
     13  purpose             200000 non-null  int64  
     14  postCode            200000 non-null  float64
     15  regionCode          200000 non-null  int64  
     16  dti                 199939 non-null  float64
     17  delinquency_2years  200000 non-null  float64
     18  ficoRangeLow        200000 non-null  float64
     19  ficoRangeHigh       200000 non-null  float64
     20  openAcc             200000 non-null  float64
     21  pubRec              200000 non-null  float64
     22  pubRecBankruptcies  199884 non-null  float64
     23  revolBal            200000 non-null  float64
     24  revolUtil           199873 non-null  float64
     25  totalAcc            200000 non-null  float64
     26  initialListStatus   200000 non-null  int64  
     27  applicationType     200000 non-null  int64  
     28  earliesCreditLine   200000 non-null  object 
     29  title               200000 non-null  float64
     30  policyCode          200000 non-null  float64
     31  n0                  189889 non-null  float64
     32  n1                  189889 non-null  float64
     33  n2                  189889 non-null  float64
     34  n3                  189889 non-null  float64
     35  n4                  191606 non-null  float64
     36  n5                  189889 non-null  float64
     37  n6                  189889 non-null  float64
     38  n7                  189889 non-null  float64
     39  n8                  189889 non-null  float64
     40  n9                  189889 non-null  float64
     41  n10                 191606 non-null  float64
     42  n11                 182425 non-null  float64
     43  n12                 189889 non-null  float64
     44  n13                 189889 non-null  float64
     45  n14                 189889 non-null  float64
    dtypes: float64(33), int64(8), object(5)
    memory usage: 70.2+ MB


通过`info()`函数，我们有三个观察：
- n0-n14都是浮点数，是匿名特征，有缺失值
- 8就业年限，16债务收入比，22公开记录清除的数量，24循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额，这些特征有缺失值

2. 描述性统计分析


```python
t.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>employmentTitle</th>
      <th>homeOwnership</th>
      <th>annualIncome</th>
      <th>verificationStatus</th>
      <th>isDefault</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>799999.000000</td>
      <td>800000.000000</td>
      <td>8.000000e+05</td>
      <td>800000.000000</td>
      <td>800000.000000</td>
      <td>...</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759729.000000</td>
      <td>759730.000000</td>
      <td>766761.000000</td>
      <td>730248.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
      <td>759730.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>399999.500000</td>
      <td>14416.818875</td>
      <td>3.482745</td>
      <td>13.238391</td>
      <td>437.947723</td>
      <td>72005.351714</td>
      <td>0.614213</td>
      <td>7.613391e+04</td>
      <td>1.009683</td>
      <td>0.199513</td>
      <td>...</td>
      <td>8.107937</td>
      <td>8.575994</td>
      <td>8.282953</td>
      <td>14.622488</td>
      <td>5.592345</td>
      <td>11.643896</td>
      <td>0.000815</td>
      <td>0.003384</td>
      <td>0.089366</td>
      <td>2.178606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>230940.252013</td>
      <td>8716.086178</td>
      <td>0.855832</td>
      <td>4.765757</td>
      <td>261.460393</td>
      <td>106585.640204</td>
      <td>0.675749</td>
      <td>6.894751e+04</td>
      <td>0.782716</td>
      <td>0.399634</td>
      <td>...</td>
      <td>4.799210</td>
      <td>7.400536</td>
      <td>4.561689</td>
      <td>8.124610</td>
      <td>3.216184</td>
      <td>5.484104</td>
      <td>0.030075</td>
      <td>0.062041</td>
      <td>0.509069</td>
      <td>1.844377</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>3.000000</td>
      <td>5.310000</td>
      <td>15.690000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>199999.750000</td>
      <td>8000.000000</td>
      <td>3.000000</td>
      <td>9.750000</td>
      <td>248.450000</td>
      <td>427.000000</td>
      <td>0.000000</td>
      <td>4.560000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>399999.500000</td>
      <td>12000.000000</td>
      <td>3.000000</td>
      <td>12.740000</td>
      <td>375.135000</td>
      <td>7755.000000</td>
      <td>1.000000</td>
      <td>6.500000e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>599999.250000</td>
      <td>20000.000000</td>
      <td>3.000000</td>
      <td>15.990000</td>
      <td>580.710000</td>
      <td>117663.500000</td>
      <td>1.000000</td>
      <td>9.000000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>10.000000</td>
      <td>19.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>799999.000000</td>
      <td>40000.000000</td>
      <td>5.000000</td>
      <td>30.990000</td>
      <td>1715.420000</td>
      <td>378351.000000</td>
      <td>5.000000</td>
      <td>1.099920e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>70.000000</td>
      <td>132.000000</td>
      <td>79.000000</td>
      <td>128.000000</td>
      <td>45.000000</td>
      <td>82.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>39.000000</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>




```python
ta.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>loanAmnt</th>
      <th>term</th>
      <th>interestRate</th>
      <th>installment</th>
      <th>employmentTitle</th>
      <th>homeOwnership</th>
      <th>annualIncome</th>
      <th>verificationStatus</th>
      <th>purpose</th>
      <th>...</th>
      <th>n5</th>
      <th>n6</th>
      <th>n7</th>
      <th>n8</th>
      <th>n9</th>
      <th>n10</th>
      <th>n11</th>
      <th>n12</th>
      <th>n13</th>
      <th>n14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>2.000000e+05</td>
      <td>200000.000000</td>
      <td>200000.000000</td>
      <td>...</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
      <td>191606.000000</td>
      <td>182425.000000</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
      <td>189889.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>899999.500000</td>
      <td>14436.954125</td>
      <td>3.481690</td>
      <td>13.244800</td>
      <td>438.737804</td>
      <td>72435.750740</td>
      <td>0.614100</td>
      <td>7.645184e+04</td>
      <td>1.010430</td>
      <td>1.744410</td>
      <td>...</td>
      <td>8.093976</td>
      <td>8.527334</td>
      <td>8.274840</td>
      <td>14.592551</td>
      <td>5.596296</td>
      <td>11.626891</td>
      <td>0.000833</td>
      <td>0.003618</td>
      <td>0.088341</td>
      <td>2.180316</td>
    </tr>
    <tr>
      <th>std</th>
      <td>57735.171256</td>
      <td>8737.430326</td>
      <td>0.855195</td>
      <td>4.766528</td>
      <td>262.246698</td>
      <td>106892.374933</td>
      <td>0.675465</td>
      <td>7.766237e+04</td>
      <td>0.781732</td>
      <td>2.367497</td>
      <td>...</td>
      <td>4.803759</td>
      <td>7.303106</td>
      <td>4.550902</td>
      <td>8.109357</td>
      <td>3.220978</td>
      <td>5.464619</td>
      <td>0.030516</td>
      <td>0.064276</td>
      <td>0.505161</td>
      <td>1.841987</td>
    </tr>
    <tr>
      <th>min</th>
      <td>800000.000000</td>
      <td>500.000000</td>
      <td>3.000000</td>
      <td>5.310000</td>
      <td>14.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>849999.750000</td>
      <td>8000.000000</td>
      <td>3.000000</td>
      <td>9.750000</td>
      <td>248.890000</td>
      <td>420.000000</td>
      <td>0.000000</td>
      <td>4.600000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>899999.500000</td>
      <td>12000.000000</td>
      <td>3.000000</td>
      <td>12.740000</td>
      <td>375.430000</td>
      <td>7836.000000</td>
      <td>1.000000</td>
      <td>6.500000e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>949999.250000</td>
      <td>20000.000000</td>
      <td>3.000000</td>
      <td>15.990000</td>
      <td>580.942500</td>
      <td>119739.250000</td>
      <td>1.000000</td>
      <td>9.000000e+04</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>10.000000</td>
      <td>19.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999999.000000</td>
      <td>40000.000000</td>
      <td>5.000000</td>
      <td>30.990000</td>
      <td>1715.420000</td>
      <td>378338.000000</td>
      <td>5.000000</td>
      <td>9.500000e+06</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>...</td>
      <td>70.000000</td>
      <td>99.000000</td>
      <td>83.000000</td>
      <td>112.000000</td>
      <td>41.000000</td>
      <td>90.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>28.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 41 columns</p>
</div>



### 2.3.3 缺失值、唯一值、异常值

#### 2.3.3.1 缺失值

缺失值的分析思路，遵循从大到小的，我们先来看一下有多少列有缺失值呢：


```python
t.isnull().any().sum()
```




    22




```python
#总样本数
len(t) 
```




    800000



然后来计算一下每个特征缺失值的占比。


```python
isnull_dict = (t.isnull().sum()/len(t))
```


```python
isnull_dict
```




    id                    0.000000
    loanAmnt              0.000000
    term                  0.000000
    interestRate          0.000000
    installment           0.000000
    grade                 0.000000
    subGrade              0.000000
    employmentTitle       0.000001
    employmentLength      0.058499
    homeOwnership         0.000000
    annualIncome          0.000000
    verificationStatus    0.000000
    issueDate             0.000000
    isDefault             0.000000
    purpose               0.000000
    postCode              0.000001
    regionCode            0.000000
    dti                   0.000299
    delinquency_2years    0.000000
    ficoRangeLow          0.000000
    ficoRangeHigh         0.000000
    openAcc               0.000000
    pubRec                0.000000
    pubRecBankruptcies    0.000506
    revolBal              0.000000
    revolUtil             0.000664
    totalAcc              0.000000
    initialListStatus     0.000000
    applicationType       0.000000
    earliesCreditLine     0.000000
    title                 0.000001
    policyCode            0.000000
    n0                    0.050338
    n1                    0.050338
    n2                    0.050338
    n3                    0.050338
    n4                    0.041549
    n5                    0.050338
    n6                    0.050338
    n7                    0.050338
    n8                    0.050339
    n9                    0.050338
    n10                   0.041549
    n11                   0.087190
    n12                   0.050338
    n13                   0.050338
    n14                   0.050338
    dtype: float64




```python
null5plus = {}
for key, value in isnull_dict.items():
    if value > .05:
        null5plus[key] = value
#打印一下结果
null5plus
```




    {'employmentLength': 0.05849875,
     'n0': 0.0503375,
     'n1': 0.0503375,
     'n2': 0.0503375,
     'n3': 0.0503375,
     'n5': 0.0503375,
     'n6': 0.0503375,
     'n7': 0.0503375,
     'n8': 0.05033875,
     'n9': 0.0503375,
     'n11': 0.08719,
     'n12': 0.0503375,
     'n13': 0.0503375,
     'n14': 0.0503375}



我们会发现，缺失值占比超过0.05的全部都为匿名特征。

再来看一下具体的分布情况，为了使图更加直观，我们需要考虑三个方面：
- 只显示缺失值
- 缺失值排序
- 柱状图


```python
isnull_dict = isnull_dict[isnull_dict>0]
isnull_dict.sort_values(inplace = True)
isnull_dict.plot.bar()
```




    <AxesSubplot:>




![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200918234935.png)


匿名特征11的缺失值比例最高，PubRecBankruptcies是缺失值最低的指标。

#### 2.3.3.2 唯一值


```python
[col for col in t.columns if t[col].nunique() <= 1]
```


    ---------------------------------------------------------------------------
    
    NameError                                 Traceback (most recent call last)
    
    <ipython-input-1-ea5c36a0e40f> in <module>
    ----> 1 [col for col in t.columns if t[col].nunique() <= 1]


    NameError: name 't' is not defined



```python

```
