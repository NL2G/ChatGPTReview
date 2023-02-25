import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import datetime as dt

sns.set_style()
sns.set_theme()


# Script to do a majority vote and analyze our annotations of Arxiv and Scholar
# Loads our annotations from "annotations/combined3.tsv" and writes the majority voted df to: data.tsv

def md_hist(df, x_name="x", y_name="y", path="save.pdf"):
    # Plot pandas 2D histogram as 3D plot
    # Code adapted from:
    # https://stackoverflow.com/questions/56336066/plotting-pandas-crosstab-dataframe-into-3d-bar-chart

    # thickness of the bars
    dx, dy = .8, .8

    # prepare 3d axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # set up positions for the bars
    xpos = np.arange(df.shape[0])
    ypos = np.arange(df.shape[1])

    # set the ticks in the middle of the bars
    ax1.set_xticks(xpos + dx / 2)
    ax1.set_yticks(ypos + dy / 2)

    # create meshgrid
    # print xpos before and after this block if not clear
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    # the bars starts from 0 attitude
    zpos = np.zeros(df.shape).flatten()

    # the bars' heights
    dz = df.values.T.ravel()

    # plot
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz)

    # put the column / index labels
    ax1.yaxis.set_ticklabels(df.columns, rotation=-15,
                             verticalalignment='center',
                             horizontalalignment='left')
    ax1.xaxis.set_ticklabels(df.index)

    # name the axes
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_zlabel('Count')

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()


# Load df and drop undefined
df = pd.read_csv("annotations/combined3.tsv", sep="\t", encoding="utf-8", index_col=False)
drop_by_topic = df[df["Topic1"].isna()]
df = df[df["Topic1"].notna()]

# Clean up ambiguous labels from annoation
df["Performance Sentiment"] = df["Performance Sentiment"].astype(int).astype(str).str.replace("0", "NAN")
df["Topic1"] = [t.split(",")[0].upper() for t in df["Topic1"].tolist()]
df["Topic1"] = ["REST" if t == "OTHER" else t for t in df["Topic1"].tolist()]
df["Topic1"] = ["ETHICS" if t in [" ETHIC", "ETHICS/REGULATIONS"] else t for t in df["Topic1"].tolist()]


def majority_vote(v, t, verbose=True):
    # Majority vote for topics/social sentiment names in t, where v are the values corresponding to t
    values = ["NAN" if pd.isna(a) else a for a in list(v)]
    values.sort()
    values = [f.upper() if type(f) == str else f for f in values]
    values = ["NAN" if pd.isna(f) else f for f in values]
    frequency = {i: values.count(i) for i in set(values)}
    frequency = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))

    # We have samples that are annotated by 1, 2 and 4 people. Here we make the decision
    if len(frequency) == 1:
        return frequency[0][0]
    e1 = frequency[0]
    e2 = frequency[1]
    if e1[1] > e2[1]:
        return e1[0]
    else:
        if verbose:
            print("Warning, unclear majority vote: ", values, " for ", t)
        return e1[0]


def majority_vote_num(v, t, verbose=True):
    # Majority vote for performance sentiment classes in t, where v are the values corresponding to t
    values = ["NAN" if pd.isna(a) else a for a in list(v)]
    values.sort()
    values = [f.upper() if type(f) == str else f for f in values]
    values = ["NAN" if pd.isna(f) else f for f in values]
    frequency = {i: values.count(i) for i in set(values)}
    frequency = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))
    if len(frequency) == 1:
        return frequency[0][0]
    e1 = frequency[0]
    e2 = frequency[1]
    # We have samples that are annotated by 1, 2 and 4 people. Here we make the decision
    if e1[0] == "NAN" and e1[1] > e2[1]:
        return e1[0]
    else:

        s = []
        for e in values:
            if e != "NAN":
                s += [float(e)]

        if verbose:
            print("Warning, unclear majority vote: ", values, " for ", t, sum(s) / len(s), str(round(sum(s) / len(s))))
        return str(round(sum(s) / len(s)))  # If nan is not the majority, we take the average of the rest


df = df.groupby("Title", as_index=False).agg({
    "Title": "first",
    "Annotator": list,
    "Venue": "first",
    "Topic1": list,
    "Topic2": list,
    "Social Sentiment": list,
    "Performance Sentiment": list,
})

# Run maj votes
df["M_Topic1"] = [majority_vote(v, t) for t, v in zip(df["Title"].tolist(), df["Topic1"].tolist())]
df["M_Topic2"] = [majority_vote(v, t, verbose=False) for t, v in zip(df["Title"].tolist(), df["Topic2"].tolist())]
df["M_Social Sentiment"] = [majority_vote(v, t) for t, v in zip(df["Title"].tolist(), df["Social Sentiment"].tolist())]
df["M_Performance Sentiment"] = [majority_vote_num(v, t) for t, v in
                                 zip(df["Title"].tolist(), df["Performance Sentiment"].tolist())]

df.to_csv("data.tsv", sep="\t")

plot = True

if plot:
    # All papers
    plt.show()
    vc = df['M_Topic1'].value_counts().sort_index(axis=0, ascending=False)
    vc.plot(kind='barh')
    print(vc)
    plt.tight_layout()
    plt.savefig("main_topic_histogram.pdf")

    plt.show()
    df['M_Topic2'].value_counts().sort_index(axis=0, ascending=False).plot(kind='barh')
    plt.tight_layout()
    plt.savefig("sub_topic_histogram.pdf")

    plt.show()
    perf = "M_Performance Sentiment"
    vc = df[perf].value_counts().sort_index(axis=0, ascending=False)
    vc.plot(kind='barh')
    print(vc)
    plt.tight_layout()
    plt.savefig("performance_sentiment.pdf")

    plt.show()
    soc = "M_Social Sentiment"
    vc = df[soc].value_counts().sort_index(axis=0, ascending=False)
    vc.plot(kind='barh')
    print(vc)
    plt.tight_layout()
    plt.savefig("social_sentiment.pdf")

    sentiment_comparison = pd.crosstab(df[perf], df[soc])
    # md_hist(sentiment_comparison, "Performance", "Social", "performance_social_cross.pdf")
    a = sns.heatmap(sentiment_comparison.T, annot=True, annot_kws={"fontsize": 8})
    a.set_xticklabels(a.get_xticklabels(), rotation=45, verticalalignment='center',
                      horizontalalignment='center')
    a.set_yticklabels(a.get_yticklabels(), rotation=0)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.set_xlabel("")
    a.set_ylabel("")
    plt.tight_layout()
    plt.savefig("heatmap_performance_social_cross.pdf")
    plt.show()

    sentiment_comparison = pd.crosstab(df[perf], df["M_Topic1"])
    # md_hist(sentiment_comparison, "Performance", "Main Topic", "performance_topic_cross.pdf")
    a = sns.heatmap(sentiment_comparison.T, annot=True, annot_kws={"fontsize": 8})
    a.set_xticklabels(a.get_xticklabels(), rotation=45, verticalalignment='center',
                      horizontalalignment='center')
    a.set_yticklabels(a.get_yticklabels(), rotation=0)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.set_xlabel("")
    a.set_ylabel("")
    plt.tight_layout()
    plt.savefig("heatmap_performance_topic_cross.pdf")
    plt.show()

    sentiment_comparison = pd.crosstab(df[soc], df["M_Topic1"])
    # md_hist(sentiment_comparison, "Social", "Main Topic", "social_topic_cross.pdf")
    a = sns.heatmap(sentiment_comparison.T, annot=True, annot_kws={"fontsize": 8})
    a.set_xticklabels(a.get_xticklabels(), rotation=45,
                      horizontalalignment='center')
    a.set_yticklabels(a.get_yticklabels(), rotation=0)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.set_xlabel("")
    a.set_ylabel("")
    plt.tight_layout()
    plt.savefig("heatmap_social_topic_cross.pdf")
    plt.show()

# ARXIV
df_arxiv = df[df["Venue"].str.lower() == "arxiv"]

# Cross time
time_df_arxiv = pd.read_csv("chatgpt.tsv", sep="\t")
time_df_arxiv = time_df_arxiv[["published", "updated", "title"]]
df_arxiv["Title"] = df_arxiv["Title"].str.lower().str.strip().replace(r'\s+', ' ', regex=True)
time_df_arxiv["title"] = time_df_arxiv["title"].str.lower().str.replace("\n", "").str.strip().replace(r'\s+', ' ',
                                                                                                      regex=True)

# Some arxiv papers were missing timestamps in the annotation, so we manually merge them in this step
secd = pd.DataFrame([["2023-02-08",
                      "A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity".lower().strip()],
                     ["2023-02-07",
                      "Reliable Natural Language Understanding with Large Language Models and Answer Set Programming".lower().strip()],
                     ["2023-02-08", "Deep Machine Learning in Cosmology: Evolution or Revolution?".lower().strip()],
                     ["2023-02-08", "Will ChatGPT get you caught? rethinking of plagiarism detection".lower().strip()],
                     ["2023-02-09",
                      "Better by you, better than me, chatgpt3 as writing assistance in students essays".lower().strip()],
                     ["2022-12-19",
                      "What would Harry say? building dialogue agents for characters in a story dialogue agents".lower().strip()],
                     ["2023-02-03",
                      "Exploring the Cognitive Dynamics of Artificial Intelligence in the Post-COVID-19 and Learning 3.0 Era: A Case Study of ChatGPT".lower().strip()]])
secd.columns = ["published2", "title"]

df_arxiv = pd.merge(df_arxiv, time_df_arxiv, how='left', left_on='Title', right_on='title')
df_arxiv = pd.merge(df_arxiv, secd, how='left', left_on='Title', right_on='title')
df_arxiv["published"] = df_arxiv["published"].fillna(df_arxiv['published2'])

if plot:
    vta = df_arxiv['M_Topic1'].value_counts().sort_index(axis=0, ascending=False)
    a = vta.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("main_topic_histogram_arxiv.pdf")

    plt.show()
    a = df_arxiv['M_Topic2'].value_counts().sort_index(axis=0, ascending=False).plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("sub_topic_histogram_arxiv.pdf")

    plt.show()
    perf = "M_Performance Sentiment"
    vpa = df_arxiv[perf].value_counts().sort_index(axis=0, ascending=False)
    a = vpa.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("performance_sentiment_arxiv.pdf")

    plt.show()
    soc = "M_Social Sentiment"
    vsa = df_arxiv[soc].value_counts().sort_index(axis=0, ascending=False)
    a = vsa.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("social_sentiment_arxiv.pdf")

    sentiment_comparison = pd.crosstab(df_arxiv[perf], df_arxiv[soc])
    md_hist(sentiment_comparison, "Performance", "Social", "performance_social_cross_arxiv.pdf")

    sentiment_comparison = pd.crosstab(df_arxiv[perf], df_arxiv["M_Topic1"])
    md_hist(sentiment_comparison, "Performance", "Main Topic", "performance_topic_cross_arxiv.pdf")

    sentiment_comparison = pd.crosstab(df_arxiv[soc], df_arxiv["M_Topic1"])
    md_hist(sentiment_comparison, "Social", "Main Topic", "social_topic_cross_arxiv.pdf")

# Scholar
df_scholar = df[df["Venue"].str.lower() != "arxiv"]
time_df_scholar = pd.read_csv("scholar_results.tsv", sep="\t")
time_df_scholar = time_df_scholar[["publicationDate", "title"]]
df_scholar["Title"] = df_scholar["Title"].str.lower().str.strip().replace(r'\s+', ' ', regex=True)
time_df_scholar["title"] = time_df_scholar["title"].str.lower().str.replace("\n", "").str.strip().replace(r'\s+', ' ',
                                                                                                          regex=True)
df_scholar = pd.merge(df_scholar, time_df_scholar, how='left', left_on='Title', right_on='title')

if plot:
    vts = df_scholar['M_Topic1'].value_counts().sort_index(axis=0, ascending=False)
    a = vts.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("main_topic_histogram_scholar.pdf")

    plt.show()
    a = df_scholar['M_Topic2'].value_counts().sort_index(axis=0, ascending=False).plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("sub_topic_histogram_scholar.pdf")

    plt.show()
    perf = "M_Performance Sentiment"
    vps = df_scholar[perf].value_counts().sort_index(axis=0, ascending=False)
    a = vps.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("performance_sentiment_scholar.pdf")

    plt.show()
    soc = "M_Social Sentiment"
    vss = df_scholar[soc].value_counts().sort_index(axis=0, ascending=False)
    a = vss.plot(kind='barh')
    a.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.savefig("social_sentiment_scholar.pdf")

    sentiment_comparison = pd.crosstab(df_scholar[perf], df_scholar[soc])
    # md_hist(sentiment_comparison, "Performance", "Social", "performance_social_cross_scholar.pdf")
    sns.heatmap(sentiment_comparison, annot=True)
    plt.tight_layout()
    plt.savefig("heatmap_performance_social_cross_scholar.pdf")
    plt.show()

    sentiment_comparison = pd.crosstab(df_scholar[perf], df_scholar["M_Topic1"])
    md_hist(sentiment_comparison, "Performance", "Main Topic", "performance_topic_cross_scholar.pdf")

    sentiment_comparison = pd.crosstab(df_scholar[soc], df_scholar["M_Topic1"])
    md_hist(sentiment_comparison, "Social", "Main Topic", "social_topic_cross_scholar.pdf")

# Recombine dataframes that have dates in order to plot them as a timeseries
dfa = pd.concat([vta, vts], axis=1).reset_index()
dfa.columns = ["Topic", "Arxiv", "SemanticScholar"]
a = dfa.plot(x="Topic", y=["SemanticScholar", "Arxiv"], kind="bar")
a.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig("topic_arxiv_scholar.pdf")
plt.show()

dfb = pd.concat([vpa, vps], axis=1).reset_index()
dfb.columns = ["Performance", "Arxiv", "SemanticScholar"]
a = dfb.plot(x="Performance", y=["SemanticScholar", "Arxiv"], kind="bar")
a.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig("performance_arxiv_scholar.pdf")
plt.show()

dfc = pd.concat([vsa, vss], axis=1).reset_index()
dfc.columns = ["Social", "Arxiv", "SemanticScholar"]
a = dfc.plot(x="Social", y=["SemanticScholar", "Arxiv"], kind="bar")
a.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
plt.savefig("social_arxiv_scholar.pdf")
plt.show()

nan = df_scholar[df_scholar["title"].isna()]
df_scholar = df_scholar[df_scholar["title"].notna()]
df_scholar["published"] = df_scholar["publicationDate"]

recombined = pd.concat([df_arxiv, df_scholar])
recombined = recombined[recombined["published"].notna()]

# Align timestamps
cleaned_dates = []
for date in recombined["published"].tolist():
    if "T" in date:
        y, m, d = date.split("T")[0].split("-")
    elif "-" in date:
        y, m, d = date.split("-")
    elif "/" in date:
        m, d, y = date.split("/")
    if len(d) == 1:
        d = "0" + d
    if len(m) == 1:
        m = "0" + m
    cleaned_dates.append(d + "." + m + "." + y)

recombined["published"] = cleaned_dates
recombined["published"] = pd.to_datetime(recombined['published'], format='%d.%m.%Y')
# Filter for timestamps in chatgpts publication period
recombined = recombined[(recombined["published"].dt.year == 2023) | (recombined["published"].dt.month == 12) | (
        recombined["published"].dt.month == 11)]

ctdf = (recombined.reset_index().groupby(["published", 'M_Topic1'], as_index=False)).count().rename(
    columns={"index": "ct"})
fig, ax = plt.subplots()

for key, data in ctdf.groupby(["M_Topic1"]):
    f = pd.date_range(start=dt.datetime(2022, 11, 30), end=dt.datetime(2023, 2, 10), freq='D')
    for date in f:
        if date not in data['published'].values:
            data = data.append(pd.Series(0, index=data.columns), ignore_index=True)
            data.loc[data.index[-1], 'published'] = date

    data = data.sort_values(by="published")
    data = data.groupby(np.arange(len(data.index)) // 7).agg({"published": "last", "ct": "sum"})
    # data["ct"] = data.rolling(2, min_periods=1).mean()["ct"]
    data.plot(x='published', y="ct", ax=ax, label=key)

plt.savefig("7daily_topic.pdf")
plt.show()

ctdf = (recombined.reset_index().groupby(["published", 'M_Performance Sentiment'], as_index=False)).count().rename(
    columns={"index": "ct"})
fig, ax = plt.subplots()

for key, data in ctdf.groupby(["M_Performance Sentiment"]):
    f = pd.date_range(start=dt.datetime(2022, 11, 30), end=dt.datetime(2023, 2, 10), freq='D')
    for date in f:
        if date not in data['published'].values:
            data = data.append(pd.Series(0, index=data.columns), ignore_index=True)
            data.loc[data.index[-1], 'published'] = date

    data = data.sort_values(by="published")
    data = data.groupby(np.arange(len(data.index)) // 7).agg({"published": "last", "ct": "sum"})
    data.plot(x='published', y="ct", ax=ax, label=key)
plt.savefig("7daily_performance.pdf")
plt.show()

ctdf = (recombined.reset_index().groupby(["published", 'M_Social Sentiment'], as_index=False)).count().rename(
    columns={"index": "ct"})
fig, ax = plt.subplots()

for key, data in ctdf.groupby(["M_Social Sentiment"]):
    f = pd.date_range(start=dt.datetime(2022, 11, 30), end=dt.datetime(2023, 2, 10), freq='D')
    for date in f:
        if date not in data['published'].values:
            data = data.append(pd.Series(0, index=data.columns), ignore_index=True)
            data.loc[data.index[-1], 'published'] = date

    data = data.sort_values(by="published")
    data = data.groupby(np.arange(len(data.index)) // 7).agg({"published": "last", "ct": "sum"})
    data.plot(x='published', y="ct", ax=ax, label=key)
plt.savefig("7daily_social.pdf")
plt.show()

# ax = sns.lineplot(data = recombined, x="published", y="")
