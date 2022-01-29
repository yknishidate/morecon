import csv
from motion import Motion


def load_csv(filepath):
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def write_csv(filepath, header, motions):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for motion in motions:
            motion.write_csv(writer)


def remove_duplicates(lst):
    return list(dict.fromkeys(lst))


def get_motion_names(csv_data):
    col = [row[0] for row in csv_data[1:]]
    return remove_duplicates(col)


def get_all_motions(csv_data):
    motions = []
    motion_names = get_motion_names(csv_data)
    for name in motion_names:
        motion = Motion(csv_data, name)
        motions.append(motion)
    return motions


def load_data(filepath):
    csv_data = load_csv(filepath)
    header = csv_data[0]
    motions = get_all_motions(csv_data)
    return motions, header
