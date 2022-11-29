"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv
import string
from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

from constants import MIMIC_3_DIR

# retain only alphanumeric
tokenizer = RegexpTokenizer(r"\w+")

char_keep = string.digits + string.ascii_lowercase + ":" + string.whitespace + "%" + "-"


def write_discharge_summaries(out_file, mode="normal"):
    notes_file = "%s/NOTEEVENTS.csv" % (MIMIC_3_DIR)
    print("processing notes file")
    with open(notes_file, "r") as csvfile:
        with open(out_file, "w") as outfile:
            spamwriter = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            print("writing to %s" % (out_file))
            outfile.write(
                ",".join(["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"]) + "\n"
            )
            notereader = csv.reader(csvfile)
            # header
            next(notereader)
            i = 0
            j = 0
            k = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    # tokenize, lowercase and remove numerics
                    if mode == "normal":
                        tokens = [
                            t.lower()
                            for t in tokenizer.tokenize(note)
                            if not t.isnumeric()
                        ]
                    elif mode == "formatted":
                        tokens = [t.lower() for t in note.split()]
                    elif mode == "diag":
                        k += 1
                        note = "".join(c for c in note.lower() if c in char_keep)
                        if "discharge diagnosis" in note:
                            j += 1
                            split = note.split("discharge diagnosis")
                            tokens = "discharge diagnoses" + split[-1]
                            tokens = [t.lower() for t in tokens.split()]
                        elif "discharge diagnoses" in note:
                            j += 1
                            tokens = (
                                "discharge diagnoses"
                                + note.split("discharge diagnoses")[-1]
                            )
                            tokens = [t.lower() for t in tokens.split()]
                        else:
                            tokens = [t.lower() for t in note.split()]
                    else:
                        raise ValueError("Unknown argument!")
                    # text = '"' + ' '.join(tokens) + '"'
                    spamwriter.writerow([line[1], line[2], line[4], " ".join(tokens)])
                i += 1
            print(j/k)
            print(k/i)
            print(j/i)
    return out_file
