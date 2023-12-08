
from pathlib import Path
import numpy as np
from tqdm import tqdm


# set these
output_dir = Path("/home/richard-rutmann/opengptx/user/richard-rutmann/data/kg_distribution/doc_indices_v2")
output_dir.mkdir(exist_ok=True)
dtype_ = np.int32
num_docs_data_1 = 29095398
num_docs_data_2 = 6455029
# num_docs_data_1 = 20
# num_docs_data_2 = 10
shuffle_datasets = True
seed = 42

doc_idx_data_1 = np.arange(0, num_docs_data_1, dtype=dtype_)
doc_idx_data_2 = np.arange(num_docs_data_1, (num_docs_data_1 + num_docs_data_2), dtype=dtype_)

if shuffle_datasets:
    np_rng = np.random.RandomState(seed=seed)
    np_rng.shuffle(doc_idx_data_1)
    np_rng.shuffle(doc_idx_data_2)


# VERSION 1: full data 1, then data 2
doc_idx_v1 = np.concatenate([doc_idx_data_1, doc_idx_data_2])
np.save(str(output_dir / f'doc_idx_v1'), doc_idx_v1)


# VERSION 2: full data 2, then data 1
doc_idx_v2 = np.concatenate([doc_idx_data_2, doc_idx_data_1])
np.save(str(output_dir / f'doc_idx_v2'), doc_idx_v2)


# VERSION 3: first half data 1, full data 2, second half data 1
cs_data_1 = len(doc_idx_data_1) // 2
doc_idx_v3 = np.concatenate([doc_idx_data_1[0:cs_data_1], doc_idx_data_2])
doc_idx_v3 = np.concatenate([doc_idx_v3, doc_idx_data_1[cs_data_1:]])
np.save(str(output_dir / f'doc_idx_v3'), doc_idx_v3)


# VERSION 4: split datasets into chunks and put them in alternating order
num_chunks = 100_000
cs_data_1 = len(doc_idx_data_1) // num_chunks
cs_data_2 = len(doc_idx_data_2) // num_chunks
doc_idx_v4 = np.array([], dtype=dtype_)
for i in tqdm(range(num_chunks), total=(num_chunks-1)):
    chunk_data_1 = doc_idx_data_1[(i*cs_data_1):(i+1)*cs_data_1]
    chunk_data_2 = doc_idx_data_2[(i * cs_data_2):(i + 1) * cs_data_2]
    doc_idx_v4 = np.concatenate([doc_idx_v4, chunk_data_1, chunk_data_2])

# add rest of datasets as last chunk
last_chunk_data_1 = doc_idx_data_1[(num_chunks*cs_data_1):]
last_chunk_data_2 = doc_idx_data_2[(num_chunks*cs_data_2):]
doc_idx_v4 = np.concatenate([doc_idx_v4, last_chunk_data_1, last_chunk_data_2])
np.save(str(output_dir / f'doc_idx_v4_nc_{num_chunks}'), doc_idx_v4)
