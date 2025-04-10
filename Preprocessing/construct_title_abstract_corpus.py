import os, json
import pandas as pd


# root_data_dir: the directory where the xls/xlsx files are stored
def load_title_abstract(root_data_dir, save_dir):
    files = os.listdir(root_data_dir)

    all_ttl_abs = []
    for cur_file in files:
        if not (cur_file.endswith('.xlsx') or cur_file.endswith('.xls')) or cur_file.startswith('.~'):
            continue 
        cur_file_full_path = os.path.join(root_data_dir, cur_file)
        cur_ttl_abs = []
        print("cur_file_full_path:", cur_file_full_path)
        if cur_file.endswith('.xlsx'):
            df = pd.read_excel(cur_file_full_path)
        elif cur_file.endswith('.xls'):
            df = pd.read_excel(cur_file_full_path, engine='xlrd')
        else:
            print(f"Unsupported file format: {cur_file}")
            continue
        # Load xls file
        df = pd.read_excel(cur_file_full_path, engine='xlrd')
        nan_values = df.isna()
        cur_titles = df['Article Title'].tolist()
        cur_abstracts = df['Abstract'].tolist()
        assert len(cur_titles) == len(cur_abstracts), "Title and Abstract lengths do not match"
        for cur_id_ttl in range(len(cur_titles)):
            if nan_values['Article Title'][cur_id_ttl] or nan_values['Abstract'][cur_id_ttl]:
                continue
            cur_ttl_abs.append([cur_titles[cur_id_ttl].strip(), cur_abstracts[cur_id_ttl].strip()])
        print("len(cur_ttl_abs):", len(cur_ttl_abs))
        all_ttl_abs.extend(cur_ttl_abs)
    print("len(all_ttl_abs):", len(all_ttl_abs))
    # get rid of repeated title-abstract pairs
    # all_ttl_abs: list of [title, abstract]
    all_ttl_abs = list(dict.fromkeys(tuple(item) for item in all_ttl_abs))
    all_ttl_abs = [list(item) for item in all_ttl_abs]
    print("len(all_ttl_abs) (no superficial repetition):", len(all_ttl_abs))
    print("all_ttl_abs[0]:", all_ttl_abs[0])

    # save to json file
    with open(save_dir, 'w') as f:
        json.dump(all_ttl_abs, f, indent=4)
    return all_ttl_abs






if __name__ == "__main__":
    root_data_dir = ""
    save_dir = ""
    load_title_abstract(root_data_dir, save_dir)