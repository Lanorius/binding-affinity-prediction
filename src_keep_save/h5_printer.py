from silx.io.dictdump import h5todict

file = "raw_data_and_data_scripts/smile_vectors_with_cids.h5"
in_file = h5todict(file)
print(in_file)
