import os



def find_files(path):
    filelist = os.listdir(path)
    for f in filelist:
        file = os.path.join(path, f)
        if os.path.isfile(file):
            if f.endswith("people"):
                label = 0
            else:
                label = 1
            # label: people 0, robot 1
            yield (file, label)


def parse_line(line):
    start = 1
    end = -1
    while line[start - 1] != "[":
        start += 1
    while line[end] != "]":
        end -= 1
    info = line[start:end].split(",")
    info = list(map(float, info))
    return info


def read_file(file):
    f = open(file[0], "r")
    line = f.readline()
    while line:
        info = parse_line(line)
        yield (info, file[1])
        # yield info
        line = f.readline()
