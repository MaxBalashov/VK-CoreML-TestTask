import json
import numpy as np
import pandas as pd
from config import HEADER, CHUNKSIZE
from config import (
    path_users_input,
    path_tracks_input,
    path_persons_input,
    path_sessions_input,
    path_users_temp,
    path_tracks_temp,
    path_persons_temp,
    path_sessions_temp
)


def str2dict(string):
    key_val_lst = [
        pair.split(':')
        for pair in string.replace('{', '').replace('}', '').replace('"', '').split(',')
    ]

    return dict(key_val_lst)


def process_users(users):
    # .apply(json.loads) здесь не парсит, используем самописный парсер
    users['properties'] = users['properties'].apply(str2dict)
    # достанем значения из 'properties'
    users['age'] = users['properties'].apply(lambda dct: dct['age'])
    users['gender'] = users['properties'].apply(lambda dct: dct['gender'])
    users['country'] = users['properties'].apply(lambda dct: dct['country'])
    users['playcount'] = users['properties'].apply(lambda dct: dct['playcount'])
    users['playlists'] = users['properties'].apply(lambda dct: dct['playlists'])
    users['user_name'] = users['properties'].apply(lambda dct: dct['lastfm_username'])
    users['subscribertype'] = users['properties'].apply(lambda dct: dct['subscribertype'])
    # выбросим лишнее
    users = users.drop(columns=[
        'type', # содержит только 'user'
        'properties' # распарсили ранее
    ])
    # заменим пропуски '' на np.nan
    users = users.replace(to_replace='', value=np.nan)
    # численные фичи приведем к численному виду
    users['age'] = users['age'].astype(float)
    users['playcount'] = users['playcount'].astype(float)
    users['playlists'] = users['playlists'].astype(float)
    return users


def process_persons(persons):
    # парсим json
    persons['properties'] = persons['properties'].apply(json.loads)
    persons['person_name'] = persons['properties'].apply(lambda dct: dct['name'])
    persons['person_MBID'] = persons['properties'].apply(lambda dct: dct['MBID'])
    # выбросим лишнее
    persons = persons.drop(columns=[
        'type', # содержит только 'person'
        'timestamp', # содержит только # -1
        'linked-entities', # содержит только '{}',
        'properties' # распарсили ранее
    ])
    return persons


def process_tracks(tracks):
    # парсим json
    tracks['properties'] = tracks['properties'].apply(json.loads)
    tracks['linked-entities'] = tracks['linked-entities'].apply(json.loads)
    # # достанем значения из 'properties'
    # tracks['track_name'] = tracks['properties'].apply(lambda dct: dct['name'])
    # tracks['track_MBID'] = tracks['properties'].apply(lambda dct: dct['MBID'])
    # tracks['duration'] = tracks['properties'].apply(lambda dct: dct['duration'])
    tracks['playcount'] = tracks['properties'].apply(lambda dct: dct['playcount'])
    # # достанем значения из 'linked-entities'
    # tracks['tags'] = tracks['linked-entities'].apply(lambda dct: dct['tags'])
    # tracks['albums'] = tracks['linked-entities'].apply(lambda dct: dct['albums'])
    tracks['person_id'] = tracks['linked-entities'].apply(lambda dct: dct['artists'][0]['id'])
    # выбросим лишнее
    tracks = tracks.drop(columns=[
        'type', # содержит только 'tracks'
        'timestamp', # содержит только # -1
        'linked-entities', # распарсили ранее
        'properties' # распарсили ранее
    ])
    return tracks


def process_sessions(sessions):
    sessions[['statistics', 'subjects']] = sessions['properties'].str.split(' ', 1, expand=True)
    # парсим json
    sessions['subjects'] = sessions['subjects'].apply(json.loads)
    sessions['statistics'] = sessions['statistics'].apply(json.loads)
    # достанем значения из 'statistics'
    sessions['playtime'] = sessions['statistics'].apply(lambda dct: dct['playtime'])
    sessions['numtracks'] = sessions['statistics'].apply(lambda dct: dct['numtracks'])
    # достанем значения из 'subjects'
    sessions['user_id'] = sessions['subjects'].apply(lambda dct: dct['subjects'][0]['id'])
    sessions['objects'] = sessions['subjects'].apply(lambda dct: dct['objects'])
    # тк в 'objects' хранится список словарей, где каждый словарь, это песня из сессии
    # то разносим каждый такой словарь в отдельную строку
    sessions = sessions.explode('objects')
    # достанем значения из 'objects'
    # также есть ('type' (содержит 'track'), 'playstart', 'playtime', 'playratio', 'action')
    sessions['track_id'] = sessions['objects'].apply(lambda dct: dct['id'])
    sessions['track_playratio'] = sessions['objects'].apply(lambda dct: dct['playratio'])
    # выбросим лишнее
    sessions = sessions.drop(columns=[
        'type', # содержит только 'event.session'
        'properties', # распарсили ранее
        'statistics', # распарсили ранее
        'subjects', # распарсили ранее
        'objects' # распарсили ранее
    ])
    return sessions


def compute_over_data(chunk_iterator, process_func):
    # df_lst = []
    # for chunk in chunk_iterator:
    #     df_lst.append(process_func(chunk))
    #     break
    return pd.concat([process_func(chunk) for chunk in chunk_iterator])


def read_data(path_data, col_names):
    # Читаем данные чанками, 
    # тк сразу полный датасет выгрузить и обработать напрямую не получается
    data = pd.read_table(
        path_data,
        header=HEADER,
        chunksize=CHUNKSIZE,
        names=col_names
    )
    return data


def save_data(data, path_data):
    data.to_csv(
        path_data,
        index=False,
        encoding='utf-8-sig'
    )



if __name__ == '__main__':

    # пользователи
    users = read_data(path_users_input, ['type', 'user_id', 'timestamp', 'properties'])
    # исполнители
    persons = read_data(path_persons_input, ['type', 'person_id', 'timestamp', 'properties', 'linked-entities'])
    # треки
    tracks = read_data(path_tracks_input, ['type', 'track_id', 'timestamp', 'properties', 'linked-entities'])
    # сессии с прослушиванием
    sessions = read_data(path_sessions_input, ['type', 'session_id', 'timestamp', 'properties'])

    # обрабатываем исходные данные по чанкам
    # .drop_duplicates(), тк допускаем, что не может быть дублей в этих таблицах
    users = compute_over_data(users, process_users).drop_duplicates()
    tracks = compute_over_data(tracks, process_tracks).drop_duplicates()
    persons = compute_over_data(persons, process_persons).drop_duplicates()
    sessions = compute_over_data(sessions, process_sessions).drop_duplicates()

    # записываем датасет
    save_data(users, path_users_temp)
    save_data(tracks, path_tracks_temp)
    save_data(persons, path_persons_temp)
    save_data(sessions, path_sessions_temp)

    print('Done!')
