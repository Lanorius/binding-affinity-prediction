from tqdm import tqdm
from statistics import stdev

from sklearn.utils import resample

import torch


def calculate_standard_error_by_bootstrapping(model_manager, test_loader, test_set, batch_size, data_used, timestamp):
    # TODO: bootstrap binary predictions

    model_manager.test(test_loader, data_used, final_prediction=True)

    # for standard deviations
    r2ms = []
    auprs = []
    cis = []

    for _ in tqdm(range(1000)):
        boot = resample(test_set, replace=True, n_samples=1000, random_state=1)
        bootloader = torch.utils.data.DataLoader(dataset=boot, batch_size=batch_size, shuffle=False)

        bootvalues_regression = model_manager.test(bootloader, data_used, bootstrap=True)
        r2ms += [bootvalues_regression[0]]
        auprs += [bootvalues_regression[1]]
        cis += [bootvalues_regression[2]]

    print("r2m std is: ", round(stdev(r2ms), 3))
    print("AUPR std is: ", round(stdev(auprs), 3))
    print("CIs std is: ", round(stdev(cis), 3))

    file1 = open("../Results/Results_"+timestamp+"/Results_"+data_used[0]+"_"+timestamp+".txt", "a")
    file1.write("\n")
    file1.write("r2m std is: "+str(round(stdev(r2ms), 3))+"\n")
    file1.write("AUPR std is: "+str(round(stdev(auprs), 3))+"\n")
    file1.write("CIs std is: "+str(round(stdev(cis), 3))+"\n")
    file1.close()
