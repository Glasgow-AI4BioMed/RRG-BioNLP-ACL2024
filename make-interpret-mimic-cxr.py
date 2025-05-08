import re
import os
import csv
from collections import defaultdict
from tqdm import tqdm
import random
import json
from random import Random
import hashlib

DICOM_VIEWS = {row["dicom_id"]: row["ViewPosition"] for row in csv.DictReader(open("mimic-cxr-2.0.0-metadata.csv"))}


def reorder_images(im_paths):
    dicoms = [os.path.basename(im).replace('.jpg', '') for im in im_paths]
    views = [DICOM_VIEWS[d] for d in dicoms]
    ranked_views = ['PA', 'AP', 'LATERAL', 'LL', 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA RLD', 'PA LLD', 'LAO', 'RAO',
                    'LPO', 'XTABLE LATERAL', 'SWIMMERS', '']
    reorder_path = []
    for r in ranked_views:
        for i, v in enumerate(views):
            if r == v:
                reorder_path.append(im_paths[i])
    return reorder_path


def main():
    print('#### mimic_cxr_sectioned.csv')
    reports = defaultdict(dict)
    for row in tqdm(csv.DictReader(open('mimic_cxr_sectioned.csv'))):

        impression = re.sub("\s+", " ", row['impression'])  # removing all line breaks
        findings = re.sub("\s+", " ", row['findings'])  # removing all line breaks

        if impression:
            reports['impression'][row['study']] = impression
        if findings:
            reports['findings'][row['study']] = findings

    print('####  mimic-cxr-2.0.0-splits.csv')

    # Grouping images per study_id
    study_images = defaultdict(list)
    for row in tqdm(csv.DictReader(open('mimic-cxr-2.0.0-split.csv'))):
        key = ('s' + row['study_id'], row['split'])
        study_images[key].append(os.path.join(
            'files',
            'p' + str(row['subject_id'])[:2],  # 10000032 -> p10
            'p' + str(row['subject_id']),
            's' + str(row['study_id']),
            row['dicom_id'] + '.jpg'
        ))

    all_samples = []
    for key, im_paths in study_images.items():
        study_id, split = key
        if split == "train":
            im_paths = reorder_images(im_paths)

            new_sample = {
                "images": im_paths,
                "images_path": im_paths,
                "impression": reports["impression"][study_id] if study_id in reports["impression"] else "",
                "findings": reports["findings"][study_id] if study_id in reports["findings"] else "",
                "source": "MIMIC-CXR",
            }
            all_samples.append(new_sample)

    val_ratio = 0.025
    num_samples = len(all_samples)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val

    indexes = list(range(num_samples))
    Random(42).shuffle(indexes)

    train_samples = [all_samples[idx] for idx in indexes[:num_train]]
    val_samples = [all_samples[idx] for idx in indexes[num_train:]]

    print("# train", len(train_samples))
    print("# val", len(val_samples))

    def save_json_and_get_hash(data, file_name):
        json_str = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        with open(file_name, "wt", encoding='utf-8') as fout:
            fout.write(json_str)
        hash_digest = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        return hash_digest

    train_hash = save_json_and_get_hash(train_samples, "train_mimic.json")
    val_hash = save_json_and_get_hash(val_samples, "val_mimic.json")
    assert train_hash == "20b1899b841efa8c53ca02ddd17bc038205b926b11b85a70b39b7c7df5aa5c3e"
    assert val_hash == "6b6421b248674d89d8b61ca5beed7db4f26b707a4c7294a34bba5b02fe38f3ae"


main()
