import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import opensmile

# Load the CSV file
df = pd.read_csv('../data/VideoTesta/VideoTestaAV__filepaths.csv')

# Rename 'Relative Path' column to 'Filename'
df = df.rename(columns={'Relative Path': 'filename'})

# Define a function to categorize the filenames and extract components
def extract_components(filename):
    pattern_sincere = r'AV_([A-Za-z]{2})([A-Za-z]{2})_([^_]*)_([^\.]*).wav'
    pattern_emotion = r'AV_([A-Za-z]{2})_([^_]*)_([^\.]*).wav'

    if re.match(pattern_sincere, filename):
        sincere, compliment, sentence, speaker = re.match(pattern_sincere, filename).groups()
        return 'sincere', sincere, compliment, sentence, speaker
    elif re.match(pattern_emotion, filename):
        emotion, sentence, speaker = re.match(pattern_emotion, filename).groups()
        return 'emotion', emotion,  None, sentence, speaker

# Apply the function and create new columns
df[['Type', 'First', 'Second', 'Sentence', 'Speaker']] = df.apply(
    lambda row: extract_components(row['filename']), axis=1, result_type='expand'
)

# Split the dataframe into two based on the filename structure
df_sincere = df[df['Type'] == 'sincere'].drop(['Type'], axis=1).sort_values(by='filename')
df_emotion = df[df['Type'] == 'emotion'].drop(['Type', 'Second'], axis=1).sort_values(by='filename')

# Reset index and rename the index column to 'id'
df_sincere = df_sincere.reset_index().rename(columns={'index': 'id'})
df_emotion = df_emotion.reset_index().rename(columns={'index': 'id'})

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

df_sincere = df_sincere.rename(columns={'First': 'sincere_label'})
df_sincere = df_sincere.rename(columns={'Second': 'critic_label'})
df_sincere = df_sincere.rename(columns={'Sentence': 'sentence_id'})
df_sincere = df_sincere.rename(columns={'Speaker': 'speaker_id'})
df_sincere['sincere_value'] = label_encoder.fit_transform(df_sincere['sincere_label'])
df_sincere['critic_value'] = label_encoder.fit_transform(df_sincere['critic_label'])

df_emotion = df_emotion.rename(columns={'First': 'emotion_label'})
df_emotion = df_emotion.rename(columns={'Sentence': 'sentence_id'})
df_emotion = df_emotion.rename(columns={'Speaker': 'speaker_id'})
df_emotion['emotion_value'] = label_encoder.fit_transform(df_emotion['emotion_label'])

# Save the DataFrames to CSV files
df_sincere.to_csv('../data/VideoTesta/fixed_irony_filepaths.csv', index=False)
df_emotion.to_csv('../data/VideoTesta/fixed_emotions_filepaths.csv', index=False)


data = np.load("../data/VideoTesta/VideoTestaAV__eGeMAPSv01b_features.npy", allow_pickle=True)
data = data.squeeze()
# Assuming the first element in each row is the filename and the rest are features
# Create a DataFrame with 'id', 'filename', and feature columns
new_df = pd.DataFrame(data)
new_df = new_df.reset_index().rename(columns={'index': 'id'})
new_df['filename'] = df['filename']

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
feature_names = smile.feature_names
new_df.columns = [new_df.columns[0]] + feature_names + ['filename']

new_df_emotions = new_df[new_df['id'].isin(df_emotion['id'])]
new_df_irony = new_df[new_df['id'].isin(df_sincere['id'])]

new_df_irony.to_csv('../data/VideoTesta/fixed_irony_VideoTestaAV__eGeMAPSv01b_features.csv', index=False)
new_df_emotions.to_csv('../data/VideoTesta/fixed_emotions_VideoTestaAV__eGeMAPSv01b_features.csv', index=False)