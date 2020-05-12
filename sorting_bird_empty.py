import shutil
import os 
import random

EXTERNAL_DRIVE_DIR = "/Volumes/Seagate Backup Plus Drive/Birds"


def copy_files(source, dest_training, dest_testing, percentage):
	src_path = os.path.join(EXTERNAL_DRIVE_DIR, source)
	dest_train_path = os.path.join(EXTERNAL_DRIVE_DIR, dest_training)
	dest_test_path = os.path.join(EXTERNAL_DRIVE_DIR, dest_testing)
	files = [file for file in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, file))]
	amount_to_train = int((percentage/100) * len(files))
	used_files = set()
	file_used = True
	word_used = ""
	if amount_to_train < 1:
		print("Nothing to collect")
	else:
		print(f"Collecting {amount_to_train} to Train")
		for i in range(0, len(files)):
			if i > amount_to_train:
				dest_path = dest_test_path
				word_used = "Testing"
			else:
				dest_path = dest_train_path
				word_used = "Training"

			file_used = True
			file = random.choice(files)
			while (file_used):
				if file in used_files:
					file = random.choice(files)
				else:
					used_files.add(file)
					file_used = False
			print(f"{i}: {word_used} Picture: {file}")
			shutil.copy(os.path.join(src_path, file), dest_path)
		print
		


def sort_training_testing():
	dates = ["2020-01-29 to 2020-02-20 DONE HH - filtered HH 06-04-20/", 
	"2020-02-20 to 2020-03-04 DONE HH - filtered HH 07-04-20/",
	"2020-03-04 to 2020-03-19 DONE KT - filtered RW 03-04-20/"]
	sorts = [("NPP/Birds", "bird"), ("NPP/Empty", "empty"), ("", "bird")]
	for date in dates:
		for path, type_file in sorts:
			src_path = os.path.join("F010 [SBH]/", date, path)
			test_path = os.path.join("Feeder10_bird_empty/testing/", type_file)
			train_path = os.path.join("Feeder10_bird_empty/training/", type_file)
			
			print(src_path)
			print(test_path)
			print(train_path)
			print("")

			copy_files(src_path, train_path, test_path, 80)

def print_file_total(path, name):
	files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
	print(f"{name}: {len(files)}")

def print_file_totals():
	src_dir = os.path.join(EXTERNAL_DRIVE_DIR, "Feeder10_bird_empty/")
	test_bird = os.path.join(src_dir, "testing/bird")
	test_empty = os.path.join(src_dir, "testing/empty")

	train_bird = os.path.join(src_dir, "training/bird")
	train_empty = os.path.join(src_dir, "training/empty")

	training_dir = os.path.join(src_dir, "training/training_data/")
	train_train_bird = os.path.join(training_dir, "train/bird")
	train_train_empty = os.path.join(training_dir, "train/empty")
	train_val_bird = os.path.join(training_dir, "val/bird")
	train_val_empty = os.path.join(training_dir, "val/empty")

	directory_names = [(train_bird, "General Training Bird"), (train_empty, "General Training Empty"),
							(test_bird, "General Testing Bird"), (test_empty, "General Testing Empty"),
							(train_train_bird, "Training (train) Bird"), (train_train_empty, "Training (train) Empty"),
							(train_val_bird, "Training (val) Bird"), (train_val_empty, "Training (val) Empty")]
	for path, name in directory_names:
		print_file_total(path, name)



def sort_testing_validation():
	src_dir = "Feeder10_bird_empty/training/"
	for type_file in ["bird", "empty"]:
		src_path = src_path = os.path.join(src_dir, type_file)
		train_path = os.path.join(src_dir, "training_data/train/", type_file)
		test_path = os.path.join(src_dir, "training_data/val/", type_file)
		
		print(src_path)
		print(train_path)
		print(test_path)
		print("")

		copy_files(src_path, train_path, test_path, 80)

def main():
	print_file_totals()
	
	
main()