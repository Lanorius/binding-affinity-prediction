from silx.io.dictdump import h5todict


comps = h5todict("smiles_embeddings_as_hd5.h5")
prots = h5todict("reduced_embeddings_T5.h5")

print(len(comps))
print(len(prots))


#print(type(in_file[in_file[0]]))
#print(len(in_file['11717001']))

