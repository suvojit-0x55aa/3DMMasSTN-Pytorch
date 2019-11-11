import sqlite3
import pandas as pd
import sys
from tqdm import tqdm


def get_file_n_face_id(db):
    return pd.read_sql_query("select file_id, face_id from Faces;", db)


def get_filepath_dims_for_image_id(db, file_id):
    query = "select filepath, width, height from FaceImages where file_id = '{}'".format(
        file_id)
    df = pd.read_sql_query(query, db)
    filepath = df.loc[0, 'filepath']
    width = df.loc[0, 'width']
    height = df.loc[0, 'height']

    return filepath, width, height


def get_landmarks(db, face_id):
    landmarks = []
    for i in range(1, 22):
        query = "SELECT feature_id, x,y from FeatureCoords WHERE face_id = '{}' AND feature_id = '{}'".format(
            face_id, i)
        df = pd.read_sql_query(query, db)

        if df.empty:
            landmarks.extend([0, 0, 0])
        else:
            landmarks.extend([df.loc[0, 'x'], df.loc[0, 'y'], 1])

    # print(landmarks)
    return landmarks


def get_face_rect(db, face_id):
    query = "select x,y,w,h from FaceRect where face_id = '{}'".format(face_id)
    df = pd.read_sql_query(query, db)

    try:
        x = df.loc[0, 'x'], df.loc[0, 'y'], df.loc[0, 'w'], df.loc[0, 'h']
    except KeyError:
        x = None

    return x


if __name__ == "__main__":
    db = sqlite3.connect(sys.argv[1])
    face_id_file_name = get_file_n_face_id(db)
    result = []

    for idx in tqdm(range(len(face_id_file_name))):
        face_id = face_id_file_name.loc[idx, 'face_id']
        file_id = face_id_file_name.loc[idx, 'file_id']

        filepath, width, height = get_filepath_dims_for_image_id(db, file_id)
        landmarks = get_landmarks(db, face_id)
        rect = get_face_rect(db, face_id)
        if rect is None:
            continue

        r = [file_id, face_id, filepath, width, height]
        r.extend(landmarks)
        r.extend(rect)
        result.append(r)

    df = pd.DataFrame(
        result,
        columns=[
            'file_id', 'face_id', 'filepath', 'im_width', 'im_height', '1_x',
            '1_y', '1_vis', '2_x', '2_y', '2_vis', '3_x', '3_y', '3_vis',
            '4_x', '4_y', '4_vis', '5_x', '5_y', '5_vis', '6_x', '6_y',
            '6_vis', '7_x', '7_y', '7_vis', '8_x', '8_y', '8_vis', '9_x',
            '9_y', '9_vis', '10_x', '10_y', '10_vis', '11_x', '11_y', '11_vis',
            '12_x', '12_y', '12_vis', '13_x', '13_y', '13_vis', '14_x', '14_y',
            '14_vis', '15_x', '15_y', '15_vis', '16_x', '16_y', '16_vis',
            '17_x', '17_y', '17_vis', '18_x', '18_y', '18_vis', '19_x', '19_y',
            '19_vis', '20_x', '20_y', '20_vis', '21_x', '21_y', '21_vis',
            'box_x', 'box_y', 'box_w', 'box_h'
        ])
    print(df.head())
    df.to_csv('aflw_processed.csv', index=False)
