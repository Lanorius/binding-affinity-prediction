from silx.io.dictdump import h5todict

file = "smiles_embeddings_as_hd5.h5"
in_file = h5todict(file)
print(len(in_file))

