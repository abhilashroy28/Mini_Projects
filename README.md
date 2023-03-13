<h2>Python Mini Projects</h2>

<h1>💪 This contains average to difficult challenges 💪</h1>

A collection of simple python mini projects I did to enhance my Python skills.

def Vader_Sentiment_Analyzer(tweet):
    tweet = SentimentIntensityAnalyzer().polarity_scores(tweet)
    return tweet


def get_compound_score(score):
    compound = score["compound"]
    return compound


def classify_vader(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    
    
Vader_after_Preprocessing = pd.DataFrame(clean_tweets["Clean_Tweets"].copy())

    
Vader_after_Preprocessing = pd.DataFrame(Vader_after_Preprocessing["Clean_Tweets"])

Vader_after_Preprocessing["Vader_SA_Score"] = Vader_after_Preprocessing[
        "Clean_Tweets"].apply(Vader_Sentiment_Analyzer)

Vader_after_Preprocessing["Compound_score"] = Vader_after_Preprocessing[
        "Vader_SA_Score"].apply(get_compound_score)

Vader_after_Preprocessing["label_by_vader"] = Vader_after_Preprocessing[
        "Compound_score"].apply(classify_vader)

positive_vs = len(Vader_after_Preprocessing[
    Vader_after_Preprocessing["label_by_vader"] ==
    "Positive"])
negative_vs = len(Vader_after_Preprocessing[
    Vader_after_Preprocessing["label_by_vader"] ==
    "Negative"])
neutral_vs = len(Vader_after_Preprocessing[
    Vader_after_Preprocessing["label_by_vader"] ==
    "Neutral"])

labels_vs = ['Positive', 'Negative', 'Neutral']
sizes_vs = [positive_vs, negative_vs, neutral_vs]
explode_vs = (0, 0.15, 0)
colors_vs = ['forestgreen', 'crimson', 'powderblue']
fig, ax = plt.subplots()
ax.pie(sizes_vs, explode=explode_vs, labels=labels_vs, colors=colors_vs, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 15})
# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')
plt.tight_layout()
plt.savefig('static/vader.png')
plt.close(fig)
