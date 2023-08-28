import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as stats


df_trans = pd.read_csv("data/transaction_test.csv")

# #perform three-way ANOVA
# model = ols("""max_score ~ C(exp_success) * C(q_short)""", data=df_trans).fit()
#
# table = sm.stats.anova_lm(model, typ=2)
# print(model.summary())
df_concrete=df_trans[df_trans['m_i']=='concrete']
df_abstract=df_trans[df_trans['m_i']=='abstract']
t_mi=stats.ttest_ind(a=df_concrete['max_score'],
                b=df_abstract['max_score'],
                equal_var=True)
print(t_mi)

t_mi_concrete=stats.ttest_ind(a=df_concrete[df_concrete['exp_success']==True]['max_score'],
                b=df_concrete[df_concrete['exp_success']==False]['max_score'],
                equal_var=True)
print(t_mi_concrete)

df_abstract=stats.ttest_ind(a=df_abstract[df_abstract['exp_success']==True]['max_score'],
                b=df_abstract[df_abstract['exp_success']==False]['max_score'],
                equal_var=True)
print(df_abstract)



fig, axes = plt.subplots(1, 2)
fig.set_size_inches(16, 9)
a=sns.boxplot(x="m_i", y="max_score", hue="exp_success", data=df_trans, palette="Set3",ax=axes[0])
a.set(xlabel="Memory item",ylabel='Max Similarity',title='Memory item comparison')
b=sns.boxplot(x="m_a", y="max_score", hue="exp_success", data=df_concrete, palette="Set3",ax=axes[1])
b.set(xlabel="Memory action",ylabel='Max Similarity',title='Memory: concrete item')
for col in axes:
    col.set_ylim([0, 1])
    col.tick_params(labelrotation=45)
    col.legend(title='Transaction?')
    col.set_xlabel('Memory action/item')
    col.set_ylabel('Max Similarity')
    col.axhline(0.6)


plt.tight_layout()
fig.savefig('data/transaction_boxplots_memory.png', dpi=800)



fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)
df_concrete=df_trans[df_trans['m_i']=='concrete']
df_success=df_concrete[df_concrete['exp_success']==True]
mean_yes=df_success.loc[:, 'max_score'].mean()
median_yes=df_success.loc[:, 'max_score'].median()
a=sns.boxplot(x="q_i", y="max_score", hue="exp_success", data=df_trans, palette="Set3",ax=axes[0,0])
a.set(xlabel="Query item",ylabel='Max Similarity',title='Memory: abstract item')
b=sns.boxplot(x="q_a", y="max_score", hue="exp_success", data=df_trans, palette="Set3",ax=axes[0,1])
b.set(xlabel="Query action",ylabel='Max Similarity',title='Memory: abstract item')
c=sns.boxplot(x="q_i", y="max_score", hue="exp_success", data=df_concrete, palette="Set3",ax=axes[1,0])
c.set(xlabel="Query item",ylabel='Max Similarity',title='Memory: concrete item')
d=sns.boxplot(x="q_a", y="max_score", hue="exp_success", data=df_concrete, palette="Set3",ax=axes[1,1])
d.set(xlabel="Query action",ylabel='Max Similarity',title='Memory: concrete item')
for row in axes:
    for col in row:
        col.set_ylim([0, 1])
        col.tick_params(labelrotation=45)
        col.legend(title='Transaction?')
        col.axhline(0.6)

plt.tight_layout()
fig.savefig('data/transaction_boxplots_query_simplified.png', dpi=800)
# plt.show()
print("test")


stats.ttest_ind(a=df_trans[df_trans['m_i']=='concrete']['max_score'],
                b=df_trans[df_trans['m_i']=='concrete']['max_score'],
                equal_var=True)