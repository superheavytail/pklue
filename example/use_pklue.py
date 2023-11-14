from pklue import get_mixture

my_hf_dataset = get_mixture(dataset_names=['kobest', 'klue'], max_examples=3000, split='train')