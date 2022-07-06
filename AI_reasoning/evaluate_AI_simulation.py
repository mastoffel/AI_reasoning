# load ai_complex_df from csv
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt

ai_complex_df = pd.read_csv('AI_reasoning/output/ai_complex_df.csv')

# subset iterations where judge > 0.5 and reason > 0.5
ai_complex_df_sub = ai_complex_df[(ai_complex_df['iter'] > 180)]
#ai_complex_df_sub = ai_complex_df[ai_complex_df['reason'] == 0.3]

# group by judge, reason and sim
ai_complex_df_sub_grouped = ai_complex_df_sub.groupby(['judge', 'reason', 'sim'])

# get mean of c1, c2, c3, c4, c5, c6, c7, c8, c9, c10
ai_complex_df_sub_grouped_mean = ai_complex_df_sub_grouped.mean()

# ungroup
plot_df = ai_complex_df_sub_grouped_mean.reset_index()

# boxplots for c10 on y and reason on x with datapoints for sim
sns.boxplot(x='reason', y='c10', data=plot_df)
sns.boxplot(x='judge', y='c10', data=plot_df)
plt.show()
sns.boxplot(x='reason', y='c10', data=ai_complex_df_sub_grouped_mean)

# lineplot for iter vs. c10, hue = sim and facetgrid by judge
sns.set(font_scale=4, style="ticks") 
g = sns.FacetGrid(ai_complex_df_sub_grouped_mean, col="judge", row ="reason", hue="sim", 
                    sharex=False, sharey=True, size=10, aspect=1.5)

# lineplot for iter vs. c10 for each judge
g.map(sns.lineplot, "iter", "c10")
g.add_legend()

# save plot to file
plt.savefig('AI_reasoning/figs/c10.png')