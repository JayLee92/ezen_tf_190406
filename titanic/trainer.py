import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ctx = r"C:/Users/ezen/PycharmProjects/test/titanic/data/"
train = pd.read_csv(ctx+"train.csv")
test = pd.read_csv(ctx+"test.csv")
#df = pd.DataFrame(train)
#print(df.columns)

#print(train.head())
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

Survival 생존여부  Survival    0 = No, 1 = Yes
Pclass 승선권 클래스   Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Sex 성별   Sex    
Age 나이   Age in years    
Sibsp 동반한 형제자매,배우자수   # of siblings / spouses aboard the Titanic    
Parch 동반한 부모, 자식 수   # of parents / children aboard the Titanic    
Ticket 티켓번호   Ticket number    
Fare 티켓의 요금   Passenger fare    
Cabin 객실번호   Cabin number    
Embarked 승선한 항구명   Port of Embarkation    
  C = Cherbourg 쉐부로, Q = Queenstown 퀸스타운, S = Southampton 사우스햄톤

"""
f, ax = plt.subplots(1, 2, figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],
                                      autopct="%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived',data=train, ax=ax[1])
ax[1].set_title('Survived')
#plt.show()

"""
데이터는 훈련데이터(train.csv) , 목적데이터(test.csv) 두가지로
제공됩니다.
목적데이터는 위 항목에서는 Survived 정보가 빠져있습니다.
그것은 답이기 때문입니다.


f, ax = plt.subplots(1, 2, figsize=(18,8))
train['Survived'][].value_counts().plot.pie(explode=[0,0.1],
                                      autopct="%1.1f%%", ax=ax[0], shadow=True)



df_1 = [train['Sex'],train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1,df_2, margins=True)
print(df.head())

"""

#train.info()

"""
위에 설명한 것처럼 0은 사망, 1은 생존을 의미합니다. 즉 탑승객의 60% 이상이 사망했다는 결론을 얻을 수 있습니다.
이번에는 남녀별 생존 비율을 확인해 보도록 하겠습니다. train_df['Survived']의 데이터에서 성별을 기준으로 필터링된 값을 가지고 비교를 해보면
"""
f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')
#plt.show()
"""
그래프 색 때문에 약간 헷갈릴 수 있지만, 남자의 사망률은 80% 이상인 반면 여자의 사망률은 약 25%정도임을 확인할 수 있습니다. 즉 lady-first가 실천되었다고 예측할 수 있습니다.
이번에는 그래프가 아닌 pandas의 자체 table 기능을 사용해서 객실 등급 데이터인 Pclass를 검토해 보겠습니다.
"""
df_1 = [train['Sex'],train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1,df_2,margins=True)
"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""
"""
이 테이블에서는 객실의 등급과 성별 별로 생존자 수를 확인할 수 있습니다. 여기에서 확인할 수 있는 정보들은…
1등 객실 여성의 생존률은 91/94 = 97%, 3등 객실 여성의 생존률은 50%
남성의 경우에 1등 객실 생존률은 37%, 3등 객실은 13%
즉 낮은 등급의 객실의 사망률이 높았다는 것으로, 좋은 자리값을 했다는 것을 볼 수 있습니다.
이번에는 ‘Embarked’, 즉 배를 탄 항구의 위치와의 연관성을 확인해 보도록 하겠습니다.
"""
f, ax = plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=train,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=train,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=train,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
#plt.show()
"""
위 데이터를 보면 절반 이상의 승객이 ‘Southampton’에서 배를 탔으며, 여기에서 탑승한 승객의 70% 가량이 남성이었습니다. 현재까지 검토한 내용으로는 남성의 사망률이 여성보다 훨씬 높았기에 자연스럽게 ‘Southampton’에서 탑승한 승객의 사망률이 높게 나왔습니다.
또한 ‘Cherbourg’에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 높은 것으로 보아서 이 동네는 부자동네라는 것을 예상할 수 있습니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show() # 예제에는 빠졌어도 넣어야 함
#bar_chart('Sex')
#bar_chart('Pclass') # 승선권 클래스
# 사망한 사람은 3등석, 생존한 사람은 1등석이 많음
#bar_chart('SibSp') # 동반한 형제자매, 배우자 수
#bar_chart('Parch') # 동반한 부모, 자식 수
#bar_chart('Embarked') # 승선한 항구명
# S, Q 에 탑승한 사람이 더 많이 사망했고 C 는 덜 사망했다

# Cabin, Ticket 값 삭제
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train.head()
test.head()
#print(train.head())
#print(test.head())

# Embarked 값 가공
#판단의 근거이기 때문에 찍어봄
#s_city = train[train['Embarked']=='S'].shape[0] #스칼라 #644
#c_city = train[train['Embarked']=='C'].shape[0] #168
#q_city = train[train['Embarked']=='Q'].shape[0] #77

#print("S = {}, C = {}, Q = {}".format(s_city,c_city,q_city))

train = train.fillna({"Embarked":"S"})
city_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)

#print(train.head())
#print(test.head())

combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])
#print(pd.crosstab(train['Title'],train['Sex']))

for dataset in combine: #카페랑 다름 카페는 Lady 두 번 들어감
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
#print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

"""
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.250000
5   Royal  1.000000
"""
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
train.head()

train = train.drop(['Name','PassengerId'],axis = 1)
test = test.drop(['Name','PassengerId'],axis = 1)
combine = [train,test]
#print(train.head)

sex_mapping = {"male":0, "female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#print(train.head())

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)
#print(train.head())

#bar_chart('AgeGroup')
"""
age_title_mapping = {1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Senior"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
train.head()

age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Adult':5,'Senior':6}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
print(train.head())

"""

age_title_mapping = {1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
train.head()

age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
#print(train.head())

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})


train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
#print(train.head())

train_data = train.drop('Survived', axis = 1)
target = train['Survived']
print(train_data.shape, target.shape)

print(train.info)