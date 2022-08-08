from cProfile import label
from humanfriendly import parse_timespan
import pandas  as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), # put the detail data
                    xy=(rect.get_x() + rect.get_width() / 2, height), # get the center location.
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def auto_text(rects):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

# 返回的是一个DataFrame数据
data = pd.read_csv("quality_result.csv", sep=",")
data = data.values.tolist()
labels = []
score = []

for rows in [data[i:i + 10] for i in range(0, len(data), 10)]:
    sort_rows = sorted(rows, key= lambda x:int(x[1]))
    for row in sort_rows:
        group = row[0].split("/")[0]
        labels.append(row[0].split("/")[1])
        score.append(int(row[1]*100)/100)

    index = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    rect1 = ax.bar(index - width / 2, score, color ='lightcoral', width=width, label =group)

    ax.set_title('Scores by gender')
    ax.set_xticks(ticks=index)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scores')

    ax.set_ylim(0, 80)

    auto_text(rect1)

    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    plt.savefig('{}.jpg'.format(str(group)), dpi=300)
    labels = []
    score = []
