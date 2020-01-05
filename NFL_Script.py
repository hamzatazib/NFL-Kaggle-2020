# Authors: Hamza Tazi Bouardi & Gabrielle Rappaport
import numpy as np
import pandas as pd
import re
from string import punctuation
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from kaggle.competitions import nflrush
from sklearn.ensemble import RandomForestRegressor
pd.options.mode.chained_assignment = None
env = nflrush.make_env()

iter_test = env.iter_test()
train_data = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
outcomes = train_data[['PlayId','Yards']].drop_duplicates().reset_index(drop=True)

### Preprocessing ###
turf_mapping = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial',
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural',
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural',
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial',
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'}


def clean_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = re.sub(' + ', ' ', txt)
    txt = txt.strip()
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('retractable', 'rtr.')
    return txt


def transform_StadiumType(txt):
    if pd.isna(txt):
        return np.nan
    if 'outdoor' in txt or 'open' in txt:
        return 1
    if 'indoor' in txt or 'closed' in txt:
        return 0

    return np.nan


def map_abbreviations(train_data):
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in train_data['PossessionTeam'].unique():
        map_abbr[abb] = abb

    train_data['PossessionTeam'] = train_data['PossessionTeam'].map(map_abbr)
    train_data['HomeTeamAbbr'] = train_data['HomeTeamAbbr'].map(map_abbr)
    train_data['VisitorTeamAbbr'] = train_data['VisitorTeamAbbr'].map(map_abbr)
    return train_data


def map_weather(txt_ini):
    try:
        txt = txt_ini.lower()
    except:
        txt = txt_ini
    weather_mapped = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        weather_mapped*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return weather_mapped * 3
    if 'sunny' in txt or 'sun' in txt:
        return weather_mapped * 2
    if 'clear' in txt:
        return weather_mapped
    if 'cloudy' in txt:
        return -weather_mapped
    if 'rain' in txt or 'rainy' in txt:
        return -2 * weather_mapped
    if 'snow' in txt:
        return -3 * weather_mapped
    return 0


def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1


def clean_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    txt = txt.replace('east', 'e')
    return txt


def transform_WindDirection(txt):
    if pd.isna(txt):
        return np.nan

    if txt=='n':
        return 0
    if txt=='nne' or txt=='nen':
        return 1/8
    if txt=='ne':
        return 2/8
    if txt=='ene' or txt=='nee':
        return 3/8
    if txt=='e':
        return 4/8
    if txt=='ese' or txt=='see':
        return 5/8
    if txt=='se':
        return 6/8
    if txt=='ses' or txt=='sse':
        return 7/8
    if txt=='s':
        return 8/8
    if txt=='ssw' or txt=='sws':
        return 9/8
    if txt=='sw':
        return 10/8
    if txt=='sww' or txt=='wsw':
        return 11/8
    if txt=='w':
        return 12/8
    if txt=='wnw' or txt=='nww':
        return 13/8
    if txt=='nw':
        return 14/8
    if txt=='nwn' or txt=='nnw':
        return 15/8
    return np.nan


def time_str_to_seconds(txt_time):
    txt_time = txt_time.split(':')
    ans = int(txt_time[0])*60 + int(txt_time[1]) + int(txt_time[2])/60
    return ans


def standardise_play_direction(df):
    #  Creating a dummy variable for plays moving left to right,
    #  and an indicator for whether or not the player is the ball carrier.
    df['ToLeft'] = (df.PlayDirection == "left") * 1

    # Convert the direction in radians
    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi / 180.0

    # Is player on offense?
    df['IsOnOffense'] = (df.Team == df.RusherTeam) * 1

    # YardLine_std variable is the line of scrimmage.
    # We ensure the YardLine_std, unlike YardLine, treats each side of the field
    # differently.
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
           'YardLine_std'
    ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
               'YardLine']

    # Change player position
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160 / 3 - df.loc[df.ToLeft, 'Y']

    # Change player direction
    df['Dir_std'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2 * np.pi)

    return df


def preprocess_data(data, test=False):
    data['StadiumTypeClean'] = data['StadiumType'].apply(clean_StadiumType)
    data['StadiumTypeClean'] = data['StadiumTypeClean'].apply(transform_StadiumType)
    data['StadiumTypeClean'] = data['StadiumTypeClean'].fillna(-1)
    data = map_abbreviations(data)
    data['HomePossesion'] = (data['PossessionTeam'] == data['HomeTeamAbbr'])*1
    data['IsRusher'] = (data['NflId'] == data['NflIdRusher'])*1
    temp = data[data["IsRusher"] == 1][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
    data = data.merge(temp, on = "PlayId")
    data["IsRusherTeam"] = (data["Team"] == data["RusherTeam"])*1
    data['GameWeatherMapped'] = data['GameWeather'].apply(lambda x: map_weather(x))
    data['TurfMapped'] = data['Turf'].map(turf_mapping)
    data['TurfMapped'] = (data['TurfMapped'] == 'Natural')*1
    data['WindSpeed'] = (data['WindSpeed']).astype(str)
    data['WindSpeedClean'] = data['WindSpeed'].apply(lambda x:
                                                  x.lower().replace('mph', '').strip() # Strip takes out the extra spaces
                                                  if not pd.isna(x)
                                                  else x)

    data['WindSpeedClean'] = data['WindSpeedClean'].apply(lambda x:
                                                  (int(x.split('-')[0])+int(x.split('-')[1]))/2 #Average WindSpeed if range
                                                  if not pd.isna(x) and '-' in x
                                                  else x)
    data['WindSpeedClean'] = data['WindSpeedClean'].apply(
        lambda x: (int(x.split()[0])+int(x.split()[-1]))/2
        if not pd.isna(x) and type(x)!=float and 'gusts up to' in x
        else x)
    data['WindSpeedClean'] = data['WindSpeedClean'].apply(str_to_float)
    data['WindSpeedClean'] = data['WindSpeedClean'].fillna(-1)
    data['WindDirectionClean'] = data['WindDirection'].apply(
        lambda x: transform_WindDirection(clean_WindDirection(x))
    )
    data['WindDirectionClean'] = data['WindDirectionClean'].fillna(-1)
    data['WindDirectionClean'].value_counts()
    # 1 if right, 0 if left
    data['PlayDirectionBool'] = data['PlayDirection'].apply(lambda x: (x.strip() == 'right')*1 )
    data['HomeField'] = (data['FieldPosition'] == data['HomeTeamAbbr'])*1
    data['YardsLeft'] = data.apply(lambda row: 100-row['YardLine']
                                               if row['HomeField'] == 1
                                               else row['YardLine'], axis=1)

    data['YardsLeft'] = data.apply(lambda row: row['YardsLeft']
                                               if row['PlayDirectionBool']
                                               else 100-row['YardsLeft'], axis=1)

    # We drop the wrong lines
    if test is False:
        data.drop(data.index[(data['YardsLeft'] < data['Yards']) |
                                         (data['YardsLeft']-100 > data['Yards'])], inplace=True)
    data['GameClockSeconds'] = data['GameClock'].apply(time_str_to_seconds)
    ## ONE HOT ENCODE OFFENSE FORMATIONS BECAUSE NO KNOWLEDGE
    data = pd.concat([data.drop(['OffenseFormation'], axis=1),
                            pd.get_dummies(data['OffenseFormation'], prefix='Formation')], axis=1)
    dummy_col = data.columns

    ## Player Height
    #We know that 1ft=12in, thus we convert to inches
    data['PlayerHeight'] = data['PlayerHeight'].apply(
        lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    ## Player BMI
    data['PlayerBMI'] = 703*(data['PlayerWeight']/(data['PlayerHeight'])**2)
    data = standardise_play_direction(data)
    #features_used = [
    #    'Quarter', 'Down', 'StadiumTypeClean',
    #    'HomePossesion', 'IsRusher', 'IsRusherTeam',
    #    'GameWeatherMapped', 'TurfMapped', 'WindSpeedClean',
    #    'WindDirectionClean', 'PlayDirectionBool', 'HomeField', 'YardsLeft',
    #    'GameClockSeconds', 'Formation_ACE', 'Formation_EMPTY',
    #    'Formation_I_FORM', 'Formation_JUMBO', 'Formation_PISTOL',
    #    'Formation_SHOTGUN', 'Formation_SINGLEBACK', 'Formation_WILDCAT',
    #    'PlayerBMI', 'ToLeft', 'Dir_rad', 'IsOnOffense', 'YardLine_std',
    #    'X_std', 'Y_std', 'Dir_std', 'Yards'
    #]
    #if test:
    #    features_used_not_in_data = list(set(features_used) - set(list(data.columns)))
    #    for col in features_used_not_in_data:
    #        data[col] = 0
    #data = data[features_used].dropna()
    print("Finished preprocessing initial dataset with feature engineering")
    return data


### Additional Preprocessing to convert into smaller dataset ###
def add_game_columns(df, test):
    columns_to_select = ['PlayId', 'Season', 'Quarter', 'Down', 'Distance', 'Week',
                         'Temperature', 'Humidity', 'StadiumTypeClean', 'GameWeatherMapped',
                         'TurfMapped', 'WindSpeedClean', 'WindDirectionClean', 'YardsLeft',
                         'GameClockSeconds', 'YardLine_std']
    if not test:
        columns_to_select.append('Yards')

    return df[columns_to_select].drop_duplicates().reset_index().drop(columns=["index"])


def team_dataframe(df):
    offense = df[df.IsOnOffense == True]
    offense = transformed_team(offense)
    offense = offense.add_prefix('Offense_')

    defense = df[(df.IsOnOffense == False)]
    defense = transformed_team(defense)
    defense = defense.add_prefix('Defense_')

    teams = pd.merge(offense, defense, left_on='Offense_PlayId', right_on='Defense_PlayId')
    teams = teams.rename(columns={"Offense_PlayId": "PlayId"}).drop(columns=['Defense_PlayId'])
    return teams


def transformed_team(df):
    columns = ['PlayId', 'Team', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel',
               'Formation_ACE', 'Formation_EMPTY', 'Formation_I_FORM', 'Formation_JUMBO',
               'Formation_PISTOL', 'Formation_SHOTGUN', 'Formation_SINGLEBACK', 'Formation_WILDCAT',
               'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'VisitorTeamAbbr', 'HomeTeamAbbr']

    set_columns_used = set(columns)
    set_columns_in_df = set(df.columns)
    columns_to_add_dummy = list(set_columns_used - set_columns_in_df)
    for col in columns_to_add_dummy:
        df[col] = 0

    df = df[columns].drop_duplicates()

    df['ScoreBeforePlay'] = df['HomeScoreBeforePlay'] * (df.Team == 'home') + df['VisitorScoreBeforePlay'] * (
                df.Team == 'away')
    df['TeamAbbr'] = df['HomeTeamAbbr'] * (df.Team == 'home') + df['VisitorTeamAbbr'] * (df.Team == 'away')

    columns = ['PlayId', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel',
               'Formation_ACE', 'Formation_EMPTY', 'Formation_I_FORM', 'Formation_JUMBO',
               'Formation_PISTOL', 'Formation_SHOTGUN', 'Formation_SINGLEBACK', 'Formation_WILDCAT',
               'ScoreBeforePlay', 'TeamAbbr']

    df = df[columns]

    return df


def generate_dataframe_for_players(df):
    offense = generate_dataframe_for_team_players(df, True)
    defense = generate_dataframe_for_team_players(df, False)
    df_1 = pd.merge(offense, defense, left_on='Offense_PlayId', right_on='Defense_PlayId')
    df_1 = df_1.rename(columns={"Offense_PlayId": "PlayId"}).drop(columns=['Defense_PlayId'])
    offense_mean = generate_mean(df, True)
    defense_mean = generate_mean(df, False)
    df_2 = pd.merge(offense_mean, defense_mean, left_on='PlayId', right_on='PlayId')
    df = pd.merge(df_1, df_2, on='PlayId')
    return df


def generate_dataframe_for_team_players(df, offense):
    offense_strategic_positions = ['OT', 'C', 'TE', 'T', 'G', 'FB', 'RB', 'HB', 'OG']
    defense_strategic_positions = ['NT', 'MLB', 'OLB', 'DL', 'LB', 'DE', 'ILB', 'DT']

    df = df[(df.IsOnOffense == offense)]
    df.X_std = df.X_std - df.YardLine_std
    columns = ['PlayId', 'Position', 'PlayerBMI', 'X_std', 'Y_std', 'S', 'A']
    df = df[columns]
    df = df.groupby(['PlayId', 'Position']).mean()
    df = df.unstack('Position')
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)

    if offense:
        columns = [k for k in df.columns if k.endswith(tuple(offense_strategic_positions))]
        columns.append("PlayId")
        df = df[columns].add_prefix('Offense_')
    else:
        columns = [k for k in df.columns if k.endswith(tuple(defense_strategic_positions))]
        columns.append("PlayId")
        df = df[columns].add_prefix('Defense_')
    return df


def generate_mean(df, offense):
    df = df[(df.IsOnOffense == offense)]
    columns = ['PlayId', 'S', 'A']
    df = df[columns]
    df = df.groupby(['PlayId']).mean()
    df.reset_index(inplace=True)

    if offense:
        df = df[columns].add_prefix('Offense_Mean_').rename(columns={"Offense_Mean_PlayId": "PlayId"})
    else:
        df = df[columns].add_prefix('Defense_Mean_').rename(columns={"Defense_Mean_PlayId": "PlayId"})
    return df


def generate_dataframe_rusher(df):
    df = df[df.IsRusher == 1]
    df.X_std = df.X_std - df.YardLine_std
    columns = ['X_std', 'Y_std', 'S', 'A', 'Position', 'PlayId']
    df = df[columns].add_prefix('Rusher_')
    df = df.rename(columns={"Rusher_PlayId": "PlayId"})
    return df


def create_df(df, test):
    initial = add_game_columns(df, test)
    players = generate_dataframe_for_players(df)
    team = team_dataframe(df)
    rusher = generate_dataframe_rusher(df)

    df_1 = pd.merge(initial, players, left_on='PlayId', right_on='PlayId')
    df_2 = pd.merge(team, rusher, left_on='PlayId', right_on='PlayId')

    df = pd.merge(df_1, df_2, left_on='PlayId', right_on='PlayId')

    return df

### Predictions & Metrics ###
def CRPS_metric(y_test, y_pred):
    y_test = np.clip(np.cumsum(y_test, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_test - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_test.shape[0])

def str_to_float_position(txt):
    try:
        return int(txt)
    except:
        return 1

train_data = preprocess_data(train_data, test=False)
print("first preprocessing done")
train_data = create_df(train_data, test=False)
print("second preprocessing done")

# Dropping these columns because strings and way too many features if OneHotEncoded
print(len(train_data.columns))
col_to_drop = ["Offense_DefensePersonnel", "Offense_OffensePersonnel", "Defense_OffensePersonnel", "Defense_DefensePersonnel"]
train_data = train_data[list(set(train_data.columns) - set(col_to_drop))]
print(len(train_data.columns))
# OneHotEncoding categorical data
train_data = pd.get_dummies(train_data)
print(train_data.shape)

X = train_data.drop(["PlayId"], axis=1).copy()
yards = X.Yards
X = X.drop("Yards", axis=1).fillna(0).reset_index(drop=True)
columns_in_train = X.columns
y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Scaled the data")

# Trainining models & obtaining Validation CRPS Scores
models = []
kf = KFold(n_splits=10, random_state=42)
score = []
for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    X_train, X_val, y_train, y_val = X[tdx], X[vdx], y[tdx], y[vdx]
    model = RandomForestRegressor(
        bootstrap=False,
        max_features=0.5,
        min_samples_leaf=20,
        max_depth=10,
        n_estimators=200,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    score_ = CRPS_metric(y_val, model.predict(X_val))
    print(score_)
    score.append(score_)
    models.append(model)
print(np.mean(score))

indices_best_scores = []
score_exp = score.copy()
for k in range(5):
    best_score_idx = np.argmin(score_exp)
    print(best_score_idx)
    indices_best_scores.append(best_score_idx)
    score_exp[best_score_idx] = 1
models_to_fit = [models[i] for i in indices_best_scores]

del X, y, X_train, X_val, y_train, y_val

# Submission to Kaggle
counter = 0
for test_df, sample_prediction_df in iter_test:
    prepro_test = preprocess_data(test_df, test=True)
    base_test = create_df(prepro_test, test=True)
    base_test.drop(["PlayId", "Defense_OffensePersonnel",
                    "Defense_DefensePersonnel", "Offense_OffensePersonnel",
                    "Offense_DefensePersonnel"], axis=1, inplace=True)
    base_test = pd.get_dummies(base_test)
    set_columns_used_in_train = set(columns_in_train)
    set_columns_in_test = set(base_test.columns)
    columns_to_add_dummy = list(set_columns_used_in_train - set_columns_in_test)
    for col in columns_to_add_dummy:
        base_test[col] = 0
    base_test = base_test.fillna(0)
    base_test = scaler.fit_transform(base_test)
    y_pred = np.mean([model.predict(base_test) for model in models_to_fit], axis=0)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)
    counter += 1
    if counter % 100 == 0:
        print(f"Predicted {counter} entries")

env.write_submission_file()
