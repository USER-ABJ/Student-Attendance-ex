import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/jenip/Downloads/drive-download-20250515T181159Z-1-001/HR_comma_sep.csv")
print(df.head())
means=df['average_montly_hours'].mean()
print(means)
df['salary']=df['salary'].astype('category')
df['sales']=df['sales'].astype('category')
model=ols('average_montly_hours ~ C(salary)+C(sales)+C(salary):C(sales)',data=df).fit()
annova=sm.stats.anova_lm(model,type=2)
print(annova)
sns.heatmap(df.groupby(['salary','sales']).mean(),annot=True)
plt.show()