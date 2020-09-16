# 萌新大赛避坑指南：天池XDatawhale零基础入门金融风控-贷款违约预测大赛

【作者】杨二茶

【日期】2020年9月15日

【声明】

1. 本指南的制作基础灵感来源于开源组织Datawhale和天池官方联合发起的0基础入门系列赛事第四场 —— 零基础入门金融风控之贷款违约预测挑战赛。
   赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。通过这道赛题来引导大家了解金融风控中的一些业务背景，解决实际问题，帮助竞赛新人进行自我练习、自我提高。

2. 之所以是避坑指南是因为本人在实践过程中过于小白，不懂如何规避各种坑，但又对于探索多种思路和路径有很多奇【keng】思【keng】妙【wa】想【wa】，由此，有了这个指南。

3. 本人学代码时间较晚，常出现玻璃心爆表和暴怒吃键盘等不稳定状态，若本文敲代码过程中，你也出现类似症状，请跟随如下指示：

吸气【我已经很棒了！】

呼气【我已经很棒了！】

吸气【我已经很棒了！】

4. 希望能帮助到和我一样毫无基础和背景的小朋友，在首次参赛的时候，少走一些弯路。

好的，做完以上的精神稳定的准备工作后，我们就可以开始第一部分**赛题理解**了~

# Task 1 赛题理解

## 0 比赛的基本信息

【比赛链接】
https://tianchi.aliyun.com/competition/entrance/531830/introduction

【参赛指南】
官方还提供了非常详细地参赛指南，详细地讲述了这个界面的使用方法，链接如下：
https://tianchi.aliyun.com/forum/postDetail?spm=5176.12281976.0.0.32ce22falA2vxS&postId=5210

【报名】

首先，我们先报名一下比赛，萌新的比赛都是个人赛，既友好也不虐汪汪。

![微信截图_20200916000617](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000806.png)

登陆阿里云以后，找到学习赛，如上图一样，点击第一个赛事，同意协议后，就能看到下图的信息了。

![微信截图_20200916000639](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000818.png)

我们可以看到比赛的状态会变成进行中，也可以加入官方交流的钉钉群，来进行讨论。当然，Datawhale的小朋友们，都有自己的学习群。

【下载数据】

下载数据需要点击**赛题与数据**，下载三个数据集到你的jupyter notebook能够索引到的目标文件夹：

- 提交样例 Sample_submit.csv
- 测试集 testA.csv
- 训练集 train.csv

![微信截图_20200916000646](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000838.png)

在做好了以上的准备后，我们终于可以开始第二步**了解学习目标**了！

## 1 学习目标

本次Task的学习目标有四：

- 理解赛题数据
- 明确目标
- 熟悉评分体系
- 熟悉比赛流程

以上的目标都太不具体了，我不知道最后要做什么！

**真.学习目标**:

- 下载数据 （已完成）
- 完成示例代码
- 提交示例结果

## 2 赛题理解

赛题理解呢，又分为四个部分，分别为：

- 赛题概况
- 数据概况
- 预测指标
- 分析赛题

下面，我们会一一道来。

### 2.1 赛题概况

【主要任务】

预测用户贷款**有没有**违约风险！

【关键词】

有没有

意味着我们可以用**零一变量**来表示**有**和**无**。

【思路】

我们的任务是希望AI最后能告诉我们，这个顾客有或者没有违约风险，从而确定要不要给他贷款，而不是相反

### 2.2 数据概况

#### 2.2.1 数据来源

某个信贷平台的贷款记录.

#### 2.2.2 数据总量

总数据量超过120w，包含47列变量信息，其中15列为匿名变量。

**【你实际分析的数据】**

- 从中抽取**80万条**作为**训练集**
- **20万条**作为**测试集A**
- **20万条**作为**测试集B**

#### 2.2.3 脱敏数据

考虑到微观数据对个人隐私的保护问题，比赛所用到的数据中有很多条目的信息都有进行脱敏处理，将这些信息用数字代替，但并不影响分析。

比赛数据详细介绍数据内容时，会略去这些包括特殊信息的**匿名特征**，因此，我们并不知道这些列的属性和特征。 

#### 2.2.4 列信息

我们的训练集所包括的列信息如下：

- id	为贷款清单分配的唯一信用证标识
- loanAmnt	贷款金额
- term	贷款期限（year）
- interestRate	贷款利率
- installment	分期付款金额
- grade	贷款等级
- subGrade	贷款等级之子级
- employmentTitle	就业职称
- employmentLength	就业年限（年）
- homeOwnership	借款人在登记时提供的房屋所有权状况
- annualIncome	年收入
- verificationStatus	验证状态
- issueDate	贷款发放的月份
- purpose	借款人在贷款申请时的贷款用途类别
- postCode	借款人在贷款申请中提供的邮政编码的前3位数字
- regionCode	地区编码
- dti	债务收入比
- delinquency_2years	借款人过去2年信用档案中逾期30天以上的违约事件数
- ficoRangeLow	借款人在贷款发放时的fico所属的下限范围
- ficoRangeHigh	借款人在贷款发放时的fico所属的上限范围
- openAcc	借款人信用档案中未结信用额度的数量
- pubRec	贬损公共记录的数量
- pubRecBankruptcies	公开记录清除的数量
- revolBal	信贷周转余额合计
- revolUtil	循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
- totalAcc	借款人信用档案中当前的信用额度总数
- initialListStatus	贷款的初始列表状态
- applicationType	表明贷款是个人申请还是与两个共同借款人的联合申请
- earliesCreditLine	借款人最早报告的信用额度开立的月份
- title	借款人提供的贷款名称
- policyCode	公开可用的策略_代码=1新产品不公开可用的策略_代码=2
- n系列匿名特征	匿名特征n0-n14，为一些贷款人行为计数特征的处理


### 2.3 预测指标

#### 2.3.1 指标定义

在经过了百度和知乎后，发现了各种各样的AUC和ROC定义如下，但可能本人较为愚笨，依然搞不明白这两者的真正意义！先把定义放在这里！

**ROC曲线**

> Receiver Operating Characteristic Curve被译为受试者工作特征曲线，根据一系列不同的二分类方法（分界值或决定阈），以真阳性率（敏感性）为纵坐标，假阳性率（1-特异性）为横坐标绘制的曲线。

但是这个指标并不能直接使用，因为ROC无法鉴别哪个分类器效果更好，因此，我们采用AUC来测评。

**AUC面积**

>**竞赛用的评价指标为AUC(Area Under Curve)**。定义为ROC曲线下与坐标轴围成的面积。

因此，AUC指标可以反应AI学习器的性能好坏，一个数值对应的更好的分类器的效果，自然是更好的。要得到AUC必须首先获得ROC。

#### 2.3.2 评测标准

AI最后生成的值是样本y的值为1的概率，这个值越大越好！


**这个地方告诉我们，AUC越大越好！冲冲冲呀！(｡･∀･)ﾉﾞ**

#### 2.3.3 分类算法的评估指标

到了这个地方，本人才粗浅大致地理解了AUC的含义和计算方法，如有更好的理解，欢迎互相交流。

1. **混淆矩阵**

先搞明白什么是混淆矩阵！而不是越看越confusing！

我们这里就一步步来，举个栗子！

**混淆矩阵的四个核心概念为**：

- positive 正类/阳性
- negative 负类/阴性
- True 真
- False 假（伪）

**【案例】**

伟大的苹果帝国缔造者（apple maker）乔布斯老先生于2011年去世了！

![微信截图_20200916000657](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000915.png)

这里，让我们先设定一下：

- 活着为阳性(+)，用**1**表示
- 死了为阴性(-)，用**0**表示

你用一个AI，预测乔布斯的生死，有以下四种情况：

- 情况一：现在是2000年，乔布斯活着(positive)，AI也预测他活着（positive），那么乔布斯就是**真.活着(True Positive)**
- 情况二：现在还是2000年，乔布斯活着（positive），AI bug了预测他死了（negative），那么，乔布斯就是**假.死了（False Negative)**
- 情况三：现在突然2020年，乔布斯死了（negative），AI也预测他死了（negative）, 那么，乔布斯就是**真.死了（True Negative）**.
- 情况四：现在还是2020年，乔布斯死了（nagative），AI又bug了预测他活着(positive)，那么，乔布斯就是**假.活着（False positive）**

这里有个小技巧，最后的真假，其实取决于AI，而不是取决于乔布斯！所以假死了和假活着，都是由AI预测结果来判断的。

最后我们在对应一下混淆矩阵：

- （1）若一个实例是正类，并且被预测为正类，即为真正类TP(True Positive )
- （2）若一个实例是正类，但是被预测为负类，即为假负类FN(False Negative )
- （3）若一个实例是负类，但是被预测为正类，即为假正类FP(False Positive )
- （4）若一个实例是负类，并且被预测为负类，即为真负类TN(True Negative )

画成图就是下面的样子：

![真假阳性](https://gitee.com/yccthu/screenshots/raw/master/img/20200915235717.png)

**2、ROC（Receiver Operating Characteristic）**

还记得ROC定义里的真阳率和假阳率吗？接下来我们在来看以下ROC的定义

【ROC空间】

- y轴：真阳率/真正例率（TPR）

TPR就是情况1和情况2中的情况，因此，总概率为$${TP + FN}$$

求法：

$$TPR = \frac{TP}{TP + FN}$$


- x轴：假阳率/假正例率（FPR）

FPR就是情况3和情况4中的情况，因此，总概率为$${FP + TN}$$

求法：

$$FPR = \frac{FP}{FP + TN}$$

下面是一张描述ROC空间的图，我们可以看到三种模型的不同的ROC曲线。

![微信截图_20200916000705](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000930.png)

在这里，我们需要重点关注一下，45°的线，**y=x**，这条线意味着真阳率和假阳率相等的情况，换成人话就是，**无论分类器把乔布斯分为死了还是活着，AI最终告诉你为活着的概率1都是均等的**。

如果要计算这时候AUC的取值，我们将横坐标和纵坐标相乘除以2，最终得到面积为0.5.

【情况一】

由图上看，如果曲线一直位于x=y线的上方，AUC的取值应该是在**[0.5,1]**,而不是**[0,1]**. 

【情况二】
如果我们的AI预测曲线恰好落于x=y上，这样就会得到一个概率为50%的预测准确度，意味着无论实际情况中乔布斯是死是活，AI预测是死和是活得概率都是50%！无偏的结果，这是一个糟糕的AI！

【情况三】
我们再想象另外一种情况，也就是曲线在x=y线的下方，取一个值x=0.4， y= 0.1，这又意味着什么呢？无论你给到AI的乔布斯是死是活，AI永远会向反方向进行预测。你得到的是一个乱说话的AI：活人被说死，死人被说话。

因此，实际情况中，我们的预测曲线，只希望为情况一！一个分类器只有达到情况一的标准，才是一个可以进行探讨预测的分类器！

**3、AUC(Area Under Curve)**

AUC（Area Under Curve）被定义为	ROC曲线	下与坐标轴围成的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。AUC越接近1.0，检测方法真实性越高;等于0.5时，则真实性最低，无应用价值。

【咸鱼一下】

到这里，我们基本上弄明白了本次比赛最为核心的指标，AUC的具体含义和计算方法了，下面开始是对总体进行的一预测。


**4、准确率（Accuracy）**

准确率是常用的一个评价指标，但是不适合样本不均衡的情况。
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**5、精确率（Precision）**

又称查准率，正确预测为正样本（TP）占预测为正样本(TP+FP)的百分比。
$$Precision = \frac{TP}{TP + FP}$$

**6、召回率（Recall）** *(同我们的真阳率！)*

又称为查全率，正确预测为正样本（TP）占正样本(TP+FN)的百分比。
$$Recall = \frac{TP}{TP + FN}$$

**7、F1 Score**

精确率和召回率是相互影响的，精确率升高则召回率下降，召回率升高则精确率下降，如果需要兼顾二者，就需要精确率、召回率的结合F1 Score。
$$F1-Score = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$

**8、P-R曲线（Precision-Recall Curve）**

P-R曲线是描述精确率和召回率变化的曲线

![微信截图_20200916000713](https://gitee.com/yccthu/screenshots/raw/master/img/20200916000957.png)



**9. KS(Kolmogorov-Smirnov)**

K-S曲线与ROC曲线类似，不同在于

- ROC曲线将真正例率和假正例率作为横纵轴
- K-S曲线将真正例率和假正例率都作为纵轴，横轴则由选定的阈值来充当。
  公式如下：
  $$KS=max(TPR-FPR)$$
  KS不同代表的不同情况，一般情况KS值越大，模型的区分能力越强，但是也不是越大模型效果就越好，如果KS过大，模型可能存在异常，所以当KS值过高可能需要检查模型是否过拟合。以下为KS值对应的模型情况，但此对应不是唯一的，只代表大致趋势。

| KS值           | 含义                               | 白话                         |
| -------------- | ---------------------------------- | ---------------------------- |
| KS值<0.2       | 一般认为模型没有区分能力           | 智障AI，可以直接放弃         |
| KS值[0.2,0.3]  | 模型具有一定区分能力，勉强可以接受 | 三岁以前的AI宝宝             |
| KS值[0.3,0.5]  | 模型具有较强的区分能力             | 稍微开始靠谱的AI宝宝         |
| KS值[0.6,0.75] | 模型具有非常强的区分能力           | 最理想的状态，可用来科研了   |
| KS值大于0.75   | 往往表示模型有异常                 | 过拟合了啊！模型太完美不现实 |

### 2.4 Baseline代码

#### 2.4.1 准备工作


```python
#基础的包包
import pandas as pd #导入pandas
import numpy as np #导入numpy
import matplotlib.pyplot as plt #导入画图用的matplotlib

#评价指标用的各种包包
from sklearn.metrics import confusion_matrix #导入混淆矩阵
from sklearn.metrics import accuracy_score #导入准确率
from sklearn import metrics #导入指标，包括精确率，召回率，F1分
from sklearn.metrics import precision_recall_curve #导入P-R曲线
from sklearn.metrics import roc_curve #导入ROC曲线
from sklearn.metrics import roc_auc_score #通过ROC来计算AUC
```


```python
t = pd.read_csv('train.csv')
ta = pd.read_csv('testA.csv') 
```

基础的观察工作如下：

1. 查看数据样式


```python
t.shape
```




    (800000, 47)




```python
ta.shape
```




    (200000, 46)




```python
t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

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
ta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

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



2. 按照类别属性观察


```python
#对象类
t.select_dtypes(include = ['object']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>subGrade</th>
      <th>employmentLength</th>
      <th>issueDate</th>
      <th>earliesCreditLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E</td>
      <td>E2</td>
      <td>2 years</td>
      <td>2014-07-01</td>
      <td>Aug-2001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D</td>
      <td>D2</td>
      <td>5 years</td>
      <td>2012-08-01</td>
      <td>May-2002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>D3</td>
      <td>8 years</td>
      <td>2015-10-01</td>
      <td>May-2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>A4</td>
      <td>10+ years</td>
      <td>2015-08-01</td>
      <td>May-1999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C</td>
      <td>C2</td>
      <td>NaN</td>
      <td>2016-03-01</td>
      <td>Aug-1977</td>
    </tr>
  </tbody>
</table>

</div>




```python
ta.select_dtypes(include = ['object']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>subGrade</th>
      <th>employmentLength</th>
      <th>issueDate</th>
      <th>earliesCreditLine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>B3</td>
      <td>10+ years</td>
      <td>2014-07-01</td>
      <td>Nov-1974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>C5</td>
      <td>10+ years</td>
      <td>2015-07-01</td>
      <td>Jul-2001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>D4</td>
      <td>2 years</td>
      <td>2016-10-01</td>
      <td>Aug-2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>C4</td>
      <td>4 years</td>
      <td>2014-11-01</td>
      <td>Jul-2002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>D1</td>
      <td>&lt; 1 year</td>
      <td>2017-10-01</td>
      <td>Dec-2000</td>
    </tr>
  </tbody>
</table>

</div>




```python
#数值类
t.select_dtypes(include = ['number']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

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
      <th>0</th>
      <td>0</td>
      <td>35000.0</td>
      <td>5</td>
      <td>19.52</td>
      <td>917.97</td>
      <td>320.0</td>
      <td>2</td>
      <td>110000.0</td>
      <td>2</td>
      <td>1</td>
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
      <td>219843.0</td>
      <td>0</td>
      <td>46000.0</td>
      <td>2</td>
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
      <td>31698.0</td>
      <td>0</td>
      <td>74000.0</td>
      <td>2</td>
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
      <td>46854.0</td>
      <td>1</td>
      <td>118000.0</td>
      <td>1</td>
      <td>0</td>
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
      <td>54.0</td>
      <td>1</td>
      <td>29000.0</td>
      <td>2</td>
      <td>0</td>
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
<p>5 rows × 42 columns</p>

</div>




```python
ta.select_dtypes(include = ['number']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


```css
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

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
      <th>0</th>
      <td>800000</td>
      <td>14000.0</td>
      <td>3</td>
      <td>10.99</td>
      <td>458.28</td>
      <td>7027.0</td>
      <td>0</td>
      <td>80000.0</td>
      <td>0</td>
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
      <td>60426.0</td>
      <td>0</td>
      <td>50000.0</td>
      <td>0</td>
      <td>2</td>
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
      <td>23547.0</td>
      <td>1</td>
      <td>60000.0</td>
      <td>2</td>
      <td>0</td>
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
      <td>636.0</td>
      <td>0</td>
      <td>37000.0</td>
      <td>1</td>
      <td>4</td>
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
      <td>368446.0</td>
      <td>1</td>
      <td>80000.0</td>
      <td>1</td>
      <td>0</td>
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
<p>5 rows × 41 columns</p>

</div>



#### 2.4.2 评价指标计算样例排雷

注意这里得样例是没有加入任何竞赛数据的，只是为了计算数值而直接给矩阵赋值，方便我们更为直观地了解每个指标运算的代码。

首先，我们先来了解一下各个指标的使用方法，如下:


```python
confusion_matrix #先简单的看一下混淆矩阵的用法
```




    <function sklearn.metrics._classification.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)>




```python
accuracy_score
```




    <function sklearn.metrics._classification.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)>




```python
metrics.precision_score
```




    <function sklearn.metrics._classification.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')>




```python
metrics.recall_score
```




    <function sklearn.metrics._classification.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')>




```python
metrics.f1_score
```




    <function sklearn.metrics._classification.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')>



然后，我们人为的制造两列只有四个数值的真实值和预测值：


```python
y_pred = [0,1,0,1] #设这个系列的预测值
y_true = [0,1,1,0] #设这个系列的真实值
```


```python
print('混淆矩阵:\n', confusion_matrix(y_true, y_pred)) #我们会得到这样的混淆矩阵！
print('ACC:',accuracy_score(y_true, y_pred))
print('Precision',metrics.precision_score(y_true, y_pred))
print('Recall',metrics.recall_score(y_true, y_pred))
print('F1-score:',metrics.f1_score(y_true, y_pred))
```

    混淆矩阵:
     [[1 1]
     [1 1]]
    ACC: 0.5
    Precision 0.5
    Recall 0.5
    F1-score: 0.5


然后，我们再来制造两列新的有10个数值的真实值和预测值，方便画图:


```python
y_pred1 = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true1 = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```


```python
%matplotlib inline
```


```python
precision_recall_curve(y_true1, y_pred1)
```




    (array([0.6       , 0.71428571, 1.        ]),
     array([1.        , 0.83333333, 0.        ]),
     array([0, 1]))




```python
plt.plot(precision, recall)
```




    [<matplotlib.lines.Line2D at 0x24096851978>]




![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200915235327.png)



```python
FPR,TPR,thresholds = roc_curve(y_true1, y_pred1)
```


```python
plt.title('ROC')
plt.plot(FPR, TPR,'b')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
```




    Text(0.5, 0, 'FPR')




![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200915235330.png)



```python
#AUC
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC socre:',roc_auc_score(y_true, y_scores))
```

    AUC socre: 0.75



```python
#KS值
FPR,TPR,thresholds=roc_curve(y_true1, y_pred1)
KS=abs(FPR-TPR).max()
print('KS值：',KS)
```

    KS值： 0.33333333333333337


## 3 赛题操作

### 3.1 赛题准备


```python
from sklearn.linear_model import LogisticRegression #逻辑回归的包
from sklearn.datasets import load_iris 
from sklearn.metrics import precision_score # 必须要为二分类
from sklearn.metrics import f1_score

```

### 3.2 赛题操作


```python
#混淆矩阵
iris = load_iris()

X,y = iris.data, iris.target

lr_clf = LogisticRegression(random_state=1)
lr_clf.fit(X,y)

y_hat = lr_clf.predict(X)

confusion_matrix(y,y_hat)
```

    d:\anaconda3\envs\python3.6\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    array([[50,  0,  0],
           [ 0, 47,  3],
           [ 0,  1, 49]], dtype=int64)



**准确率**


```python
#准确率
score = accuracy_score(y,y_hat)
accuracy_score(y,y_hat,normalize=False),int(150 * score)
```




    (146, 146)



**精确率**


```python
precision_score(y, y_hat,average=None)
```




    array([1.        , 0.97916667, 0.94230769])




```python
precision_score(y,y_hat, average='macro')
```




    0.9738247863247862




```python
precision_score(y,y_hat, average='micro')
```




    0.9733333333333334




```python
np.mean(precision_score(y, y_hat,average=None)) # 等同于 macro模 式
```




    0.9738247863247862




```python
f1_score(y, y_hat, average=None)
```




    array([1.        , 0.95918367, 0.96078431])



**P-R曲线**


```python
y_score = lr_clf.decision_function(X) # 打印出样本的置信度
```


```python
from sklearn.metrics import average_precision_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores)
```




    0.8333333333333333




```python
from sklearn.metrics import precision_recall_curve

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision,recall,thresholds
```




    (array([0.66666667, 0.5       , 1.        , 1.        ]),
     array([1. , 0.5, 0.5, 0. ]),
     array([0.35, 0.4 , 0.8 ]))




```python
# 示例
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


X,y = make_classification(n_samples=200, n_features=20,n_informative=2, n_classes=2,shuffle=True)

print(X.shape, y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, shuffle=False)

svc = LinearSVC(random_state=2)
svc.fit(X_train,y_train)

y_score = svc.decision_function(X_test)   # 搞懂这个是啥意思
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

disp = plot_precision_recall_curve(svc,X_test, y_test)

disp.ax_.set_title('2-class Precision-Recall curve：AP={0:0.2f}'.format(average_precision))
```

    (200, 20) (200,)
    Average precision-recall score: 0.92


    d:\anaconda3\envs\python3.6\lib\site-packages\sklearn\svm\_base.py:977: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    Text(0.5, 1.0, '2-class Precision-Recall curve：AP=0.92')



    d:\anaconda3\envs\python3.6\lib\site-packages\matplotlib\backends\backend_agg.py:238: RuntimeWarning: Glyph 65306 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    d:\anaconda3\envs\python3.6\lib\site-packages\matplotlib\backends\backend_agg.py:201: RuntimeWarning: Glyph 65306 missing from current font.
      font.set_text(s, 0, flags=flags)



![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200915235337.png)


**ROC**


```python
from sklearn.metrics import roc_auc_score,roc_curve

roc_auc_score(y_test,y_score,average=None)
```




    0.9245714285714286




```python
fpr,tpr,thresholds = roc_curve(y_test, y_score)
```


```python
from sklearn.metrics import plot_roc_curve

plot_roc_curve(svc,X_test,y_test,drop_intermediate=False)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x24098b287b8>




![png](https://gitee.com/yccthu/screenshots/raw/master/img/20200915235339.png)


**AUC**


```python
from sklearn.metrics import auc
auc(fpr,tpr)
```




    0.9245714285714286




```python

```

### 4 经验总结

【数据提交要求】

- 只可提交2次
- 结果csv格式
- 大小为10M以内

![微信截图_20200916000726](https://gitee.com/yccthu/screenshots/raw/master/img/20200916001211.png)

【今日小结】

今天把基础代码全部过了一遍，既有较为简单的操作，也有比较复杂的，一些地方还需要进一步理解，不过敲完代码就是很好的开始了呢！加油！(｡･∀･)ﾉﾞ

### 5. 评分卡小技巧

评分卡是一张拥有分数刻度会让相应阈值的表。信用评分卡是用于用户信用的一张刻度表。以下代码是一个非标准评分卡的代码流程，用于刻画用户的信用评分。评分卡是金融风控中常用的一种对于用户信用进行刻画的手段哦！


```python
def Score(prob,P0=600,PDO=20,badrate=None,goodrate=None):
    P0 = P0
    PDO = PDO
    theta0 = badrate/goodrate
    B = PDO/np.log(2)
    A = P0 + B*np.log(2*theta0)
    score = A-B*np.log(prob/(1-prob))
    return score
```


```python
Score
```




    <function __main__.Score(prob, P0=600, PDO=20, badrate=None, goodrate=None)>

