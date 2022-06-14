import pandas as pd
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import plotly.express as px


def getBestGenres(data, min):
    dictionaryOfGenres = {}

    for x in range(len(data)):
        genres = data.loc[x, 'artist_genres']
        genres = genres[1:len(genres) - 1].split(",")

        for a in range(len(genres)):
            genre = genres[a].strip()
            genre = genre[1:len(genre) - 1]
            if genre in dictionaryOfGenres:
                dictionaryOfGenres[genre] = dictionaryOfGenres[genre] + 1
            else:
                dictionaryOfGenres[genre] = 1

    t = sorted(dictionaryOfGenres.items(), key=lambda x: x[1], reverse=True)

    listOfGenres = []
    for a in t:
        if a[1] > min:
            print(a[0], a[1])
            listOfGenres.append(a[0])

    return listOfGenres


def addGenresToColumns(data, listOfGenres):
    for x in range(len(data)):
        genres = data.loc[x, 'artist_genres']
        genres = genres[1:len(genres) - 1].split(",")

        l = []

        for a in range(len(genres)):
            genre = genres[a].strip()
            genre = genre[1:len(genre) - 1]
            l.append(genre)

        for g in listOfGenres:
            if g in l:
                data.loc[x,g] = 1
            else:
                data.loc[x, g] = 0

    return data


def showBoxplot(dataFrame, listOfColumns):
    for a in listOfColumns:
        fig = px.box(dataFrame, y=a)
        fig.show()


def getOutliers(dataFrame, column):
    q1 = dataFrame[column].quantile(0.25)
    q3 = dataFrame[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return dataFrame.index[(dataFrame[column] < lower) | (dataFrame[column] > upper)]


def removeOutliers(dataFrame, indexes):
    indexes = sorted(set(indexes))
    return dataFrame.drop(indexes)

def getDataFrameWithoutOutliers(dataFrame, columns):
    indexList = []
    for f in columns:
        indexList.extend(getOutliers(dataFrame, f))

    return removeOutliers(dataFrame, indexList)




pd.set_option('display.expand_frame_repr', False)


trainData = pd.read_csv("spotify_train.csv")
testData = pd.read_csv("spotify_test.csv")


# kodujem rok v trénovacích dátach
for x in range(len(trainData)):
  trainData.loc[x, 'release_date'] = int(trainData.loc[x, 'release_date'].split("-")[0])
trainData["release_date"] = pd.to_numeric(trainData["release_date"])

# kodujem rok v testovacích dátach
for x in range(len(testData)):
  testData.loc[x, 'release_date'] = int(testData.loc[x, 'release_date'].split("-")[0])
testData["release_date"] = pd.to_numeric(testData["release_date"])

# # Vyberám také žanre pesničiek ktoré sa v datasete spotify_train nachádzajú minimálne 1500 krát
genresListColumns = getBestGenres(trainData, 1500)


# kodujem žánre v trenovacích dátach
trainData = addGenresToColumns(trainData, genresListColumns)


# kodujem žánre v testovacich dátach
testData = addGenresToColumns(testData, genresListColumns)


# Odstráňujem duplicity v trénovacich dátach
trainData.sort_values("popularity", ascending=False, inplace=True)
trainData.drop_duplicates(subset=["artist_id", "name"], keep="first", inplace=True)


# Mažem nepotrebné stlpce
trainData = trainData.drop(["id", "artist_id", "artist", "name", "explicit", "url", "playlist_id",
                            "playlist_description", "playlist_name", "playlist_url", "query", "artist_genres",
                            "popularity", "key", "mode", "artist_followers", "duration_ms", "speechiness", "liveness",
                            "tempo"], axis=1)

testData = testData.drop(["id", "artist_id", "artist", "name", "explicit", "url", "playlist_id",
                          "playlist_description", "playlist_name", "playlist_url", "query", "artist_genres",
                          "popularity", "key", "mode", "artist_followers", "duration_ms", "speechiness", "liveness",
                          "tempo"], axis=1)


trainData = trainData.dropna()

# Odstraňujem také riadky v trénovacích dátach kde loudness hodnota je väčšia ako 0 a menšie ako -60
trainData = trainData.drop(trainData[trainData.loudness > 0].index)
trainData = trainData.drop(trainData[trainData.loudness < -60].index)

# Odstránenie oulierov
columns = ["release_date", "danceability", "energy", "acousticness", "instrumentalness", "valence"]
showBoxplot(trainData, columns)
trainData = getDataFrameWithoutOutliers(trainData, columns)


trainData.to_csv("spotify_train_prepared_data.csv")
testData.to_csv("spotify_test_prepared_data.csv")
print(trainData)
print(testData)


print(trainData)

trainData = pd.read_csv("spotify_train_prepared_data.csv")
testData = pd.read_csv("spotify_test_prepared_data.csv")


# Normalizácia dat
scaler = MinMaxScaler().fit(trainData)
trainData = pd.DataFrame(scaler.transform(trainData), columns=trainData.columns)
testData = pd.DataFrame(scaler.transform(testData), columns=testData.columns)


trainDataY = trainData["loudness"]
trainDataX = trainData.drop(['loudness'], axis=1)

testDataY = testData["loudness"]
testDataX = testData.drop(['loudness'], axis=1)


# Trénovacie dáta
classifier = SVR(verbose=True)
a = classifier.fit(trainDataX, trainDataY)

# Krížová validácia
cross = cross_val_score(classifier, trainDataX, trainDataY, cv=None)
print("Križová validácia")
print(cross)
print("---------------------------------------------------------------------------")

yPredTrain = classifier.predict(trainDataX)
yPredTest = classifier.predict(testDataX)


print("MSE - testovacie dáta:")
print(mean_squared_error(testDataY.values, yPredTest))
print("R2 - testovacie dáta:")
print(r2_score(testDataY.values, yPredTest))

print("MSE - trénovacie dáta:")
print(mean_squared_error(trainDataY.values, yPredTrain))
print("R2 - trénovacie dáta:")
print(r2_score(trainDataY.values, yPredTrain))

print("---------------------------------------------------------------------------")


forResidualPlot = {'prediction': yPredTest, 'residual': yPredTest - testDataY}

fig = px.scatter(
    forResidualPlot, x='prediction', y='residual',
    marginal_y='violin',
    trendline='ols'
)
fig.show()

# GridSearch
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100]}
gridSearch = GridSearchCV(classifier, parameters, verbose=1)
gridSearch.fit(trainDataX, trainDataY)

gridPred = gridSearch.predict(testDataX)

print("gridSearch predict:")
print(gridPred)
print("MSE - GridSearch - testovacie dáta:")
print(mean_squared_error(testDataY.values, gridPred))
print("R2 - GridSearch - testovacie dáta:")
print(r2_score(testDataY.values, gridPred))
print("gridSearch najlepšie parametre ")
print(gridSearch.best_params_)

print("---------------------------------------------------------------------------")


# BAGGING
bagging = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
bagging.fit(trainDataX, trainDataY)
baggingPred = bagging.predict(testDataX)

print("bagging regressor predict:")
print(baggingPred)
print("MSE - bagging regressor predict:")
print(mean_squared_error(testDataY.values, baggingPred))
print("R2 - bagging regressor predict:")
print(r2_score(testDataY.values, baggingPred))


print("---------------------------------------------------------------------------")


# BOOST
# boost = AdaBoostRegressor(learning_rate=0.5, loss='exponential', random_state=0, n_estimators=100)
boost = AdaBoostRegressor(random_state=0, n_estimators=100)
boost.fit(trainDataX, trainDataY)
boostPred = boost.predict(testDataX)

print("boost regressor predict:")
print(boostPred)
print("MSE - boost regressor predict:")
print(mean_squared_error(testDataY.values, boostPred))
print("R2 - boost regressor predict:")
print(r2_score(testDataY.values, boostPred))

