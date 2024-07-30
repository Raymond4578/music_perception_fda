

from dataloader import load_data

target_dict, data_dict = load_data('./data/')
for key in target_dict:
    print(key)
    print(target_dict[key])

for key in data_dict:
    print(key)
    print(data_dict[key])