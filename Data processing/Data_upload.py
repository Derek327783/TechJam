from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="../../folder")
dataset.push_to_hub("Monke64/Music2Image",private=True)

