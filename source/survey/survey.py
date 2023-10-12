from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import re
from wordcloud import WordCloud
# import CSV file
data = pd.read_csv('survey.csv', delimiter=',',dtype='str,str,str', names=["time","bad", "good"],header=None)
g_words = ["hw", "lec style/cont", "theory/rigor", "nothing", "Instructor OH", "Zoom option", "SVM", "team work", "coding", "Taking feedback"]
b_words = ["hw len/diff", "hw support", "theory/rigor", "in-class eg", "lec pace", "hw clarity", "hw/lec conn", "lec notes", "extra cred hw", "feedback", "review/preview", "lec style/cont", "lec detail", "workload","More OH", "visual"]

num_good = np.zeros(len(g_words),dtype=int)
num_bad = np.zeros(len(b_words),dtype=int)

# hw len/diff
num_good[0] = 5 
num_bad[0] = 13

# hw support
num_bad[1] = 9

# lecture style/contents
num_good[1] = 17
num_bad[11] = 2

# theory/rigor
num_good[2] = 11
num_bad[2] = 3

# nothing
num_good[3] = 4

# Instructor OH
num_good[4] = 9

# examples in class
num_bad[3] = 9

# lec pace
num_bad[4] = 3

# hw clarity
num_bad[5] = 13

# Zoom option
num_good[5] = 2

# SVM
num_good[6] = 1

# hw/lec conn
num_bad[6] = 1

# lec notes
num_bad[7] = 3

# team work
num_good[7] = 1

# extra cred hw
num_bad[8] = 2

# feedback
num_bad[9] = 1
num_good[9] = 4

# coding
num_good[8] = 4

# review/preview
num_bad[10] = 2

# lec detail
num_bad[12] = 1

# workload
num_bad[13] = 2

# more OH
num_bad[14] = 1

# visual
num_bad[15] = 1
# arrange in descending order
ind_good = np.argsort(-num_good)
ind_bad = np.argsort(-num_bad)
g_words = np.array(g_words)[ind_good]
b_words = np.array(b_words)[ind_bad]
num_good = num_good[ind_good]
num_bad = num_bad[ind_bad]

"""
def find_gist(text):
    # split text into words
    count = np.zeros(len(words),dtype=int)
    for i in range(len(words)):
        if re.search(words[i], text, re.IGNORECASE):
            count[i] = 1
    return count
# find gist of each comment
good_gist = np.zeros((len(data), len(words)),dtype=int)
bad_gist = np.zeros((len(data), len(words)),dtype=int)
for i in range(len(data)):
    good_gist[i] = find_gist(data["good"][i])
    bad_gist[i] = find_gist(data["bad"][i])
"""


# plot wordcloud
good_text = " ".join(data["good"])
bad_text = " ".join(data["bad"])
wordcloud = WordCloud().generate(good_text)
wordcloud2 = WordCloud().generate(bad_text)
fig2, ax2 = subplots()
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Favorite aspect',fontsize=18)
fig3, ax3 = subplots()
ax3.imshow(wordcloud2, interpolation='bilinear')
ax3.axis("off")
ax3.set_title('Needs improvement',fontsize=18)




fig, ax = subplots()
fig1, ax1 = subplots()
#num_good = np.sum(good_gist, axis=0)
#num_bad = np.sum(bad_gist, axis=0)
ax.barh(g_words, num_good, align='center')
ax1.barh(b_words, num_bad, align='center')
ax.set_title('Favorite aspect',fontsize=18)
ax1.set_title('Needs improvement',fontsize=18)
ax.set_xlabel('Number of occurrences',fontsize=18)
ax1.set_xlabel('Number of occurrences',fontsize=18)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)
tight_layout()
show()

