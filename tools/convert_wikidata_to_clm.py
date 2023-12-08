
from datetime import datetime
import json
from pathlib import Path


chunk_size = 10_000_000
# chunk_size = 10
translated_wikidata_path = Path("/home/richard-rutmann/s3/data/wikidata/translated_wikidata-20220103-all_lut_v1.jsonl")
output_dir = Path("/home/richard-rutmann/s3/data/wikidata/")

output_file = output_dir / f"clm_{translated_wikidata_path.name.replace('translated_', '')}"
output_file.unlink(missing_ok=True)


def save_wikidata(clm_wikidata):
    with output_file.open('a') as f:
        for entry in clm_wikidata:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + "\n")


def print_with_time(msg: str):
    ts = datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
    print(ts + " " + msg)


# translate IDs in Wikidata LUT
print_with_time("Convert triples to CLM")
clm_triples = []
num_clm_triples_chunk = 0
num_clm_triples = 0

with translated_wikidata_path.open('r') as f:
    for i, line in enumerate(f.readlines()):
        subgraph = json.loads(line)['subgraph']

        for (subj, rel, obj) in subgraph:
            clm_triples.extend([
                {"text": f"If the subject is {subj} and the relation is {rel}, the object is {obj}"},
                {"text": f"If the object is {obj} and the relation is {rel}, the subject is {subj}"},
                {"text": f"If the subject is {subj} and the object is {obj}, the relation is {rel}"}
            ])
            num_clm_triples_chunk += 3

        if num_clm_triples_chunk >= chunk_size:
            print_with_time(f"Processed {i} entries in Wikidata")
            save_wikidata(clm_triples)
            clm_triples = []
            num_clm_triples += num_clm_triples_chunk
            num_clm_triples_chunk = 0

# save
num_clm_triples += num_clm_triples_chunk
save_wikidata(clm_triples)
print_with_time(f"{num_clm_triples=}")
print_with_time("Done")
