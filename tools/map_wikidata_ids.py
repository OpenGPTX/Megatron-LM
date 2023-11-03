
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm


chunk_size = 1_000_000
wikidata_full_lut_path = Path("/home/richard-rutmann/s3/data/wikidata/wikidata-20220103-all_lut.jsonl")
output_dir = Path("/home/richard-rutmann/s3/data/wikidata/")

output_file = output_dir / f"translated_{wikidata_full_lut_path.name}"
entities_not_found_file = output_dir / "entities_not_found.txt"
output_file.unlink(missing_ok=True)
entities_not_found_file.unlink(missing_ok=True)


def save_wikidata(translated_wikidata):
    print_with_time(f"Write chunk of translated triples to output file")
    with output_file.open('a') as f:
        for entry in translated_wikidata:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + "\n")


def print_with_time(msg: str):
    ts = datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
    print(ts + " " + msg)


# read wikidata LUT
print_with_time("Read Wikidata LUT")
wikidata_full_lut = {}
with wikidata_full_lut_path.open('r') as f:
    for line in f.readlines():
        d = json.loads(line)
        wikidata_full_lut[d['wd_id']] = d


# translate IDs in Wikidata LUT
print_with_time("Translate IDs in Wikidata LUT")
translated_wikidata = []
num_ignored_triples = 0
num_translated_triples = 0
entities_not_found = set()
for wd_id, wd_entry in tqdm(wikidata_full_lut.items()):
    translated_wikidata_entry = {k: v for k, v in wd_entry.items() if k not in ('subgraph', 'neighbour_ids')}
    translated_wikidata_entry['subgraph'] = []
    translated_wikidata_entry['instance_of'] = ""
    for triple in wd_entry['subgraph']:
        translated_triple = []
        try:
            for entity_id in triple:
                translated_triple.append(wikidata_full_lut[entity_id]["labels"]["value"])
            if triple[1] == 'P31' and triple[0] == wd_id:  # instance of
                translated_wikidata_entry['instance_of'] = triple[-1]
        except KeyError:
            num_ignored_triples += 1
            entities_not_found.add(entity_id)
            continue
        else:
            translated_wikidata_entry['subgraph'].append(translated_triple)
            num_translated_triples += 1

    translated_wikidata.append(translated_wikidata_entry)

    if num_translated_triples >= chunk_size:
        save_wikidata(translated_wikidata)
        translated_wikidata = []
        num_translated_triples = 0

# save
print_with_time(f"Save to {output_file}")
save_wikidata(translated_wikidata)
print_with_time(f"{num_ignored_triples=}")
print_with_time(f'Write not found entities to {entities_not_found_file}')
with entities_not_found_file.open('w') as f:
    f.write("\n".join(entities_not_found))

print_with_time("Done")
