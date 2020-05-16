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
		


def sort_training_testing_bird_empty():
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

def sort_train_test_bird_parakeet():
	dates = ["2020-01-29 to 2020-02-20 DONE HH - filtered HH 06-04-20/", 
	"2020-02-20 to 2020-03-04 DONE HH - filtered HH 07-04-20/",
	"2020-03-04 to 2020-03-19 DONE KT - filtered RW 03-04-20/"]
	sorts = [("NPP/Birds", "bird"), ("", "parakeet")]
	for date in dates:
		for path, type_file in sorts:
			src_path = os.path.join("F010 [SBH]/", date, path)
			test_path = os.path.join("Feeder10_bird_parakeet/testing/", type_file)
			train_path = os.path.join("Feeder10_bird_parakeet/training/", type_file)
			print(src_path)
			print(test_path)
			print(train_path)
			print("")

			copy_files(src_path, train_path, test_path, 80)

def sort_training_testing(dates, dest_path, feeder, sorts):
	
	for date in dates:
		for path, type_file in sorts:
			src_path = os.path.join(feeder, date, path)
			test_path = os.path.join(dest_path, "testing/", type_file)
			train_path = os.path.join(dest_path, "training/", type_file)
			print(src_path)
			print(test_path)
			print(train_path)
			print("")

			copy_files(src_path, train_path, test_path, 80)


def print_file_total(path, name):
	files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
	print(f"{name}: {len(files)}")

def print_file_totals(directory, type_files):
	src_dir = os.path.join(EXTERNAL_DRIVE_DIR, directory)
	for file_type in type_files:
		test_path = os.path.join(src_dir, "testing/", file_type)
		print_file_total(test_path, f"Testing {file_type}")
		train_path = os.path.join(src_dir, "training/", file_type)
		print_file_total(train_path, f"Training {file_type}")
		training_dir = os.path.join(src_dir, "training/training_data/")
		train_train = os.path.join(training_dir, "train/", file_type)
		print_file_total(train_train, f"Training (train) {file_type}")
		train_val = os.path.join(training_dir, "val/", file_type)
		print_file_total(train_val, f"Training (val) {file_type}")



def sort_testing_validation(src_dir, type_files):
	for type_file in type_files:
		src_path = os.path.join(src_dir, type_file)
		train_path = os.path.join(src_dir, "training_data/train/", type_file)
		test_path = os.path.join(src_dir, "training_data/val/", type_file)
		
		print(src_path)
		print(train_path)
		print(test_path)
		print("")

		copy_files(src_path, train_path, test_path, 80)

def print_directory_totals(ext_dir, list_dirs):
	print(ext_dir)
	for path, type_files in list_dirs:
		print(path)
		print_dir = os.path.join(ext_dir, path)
		print_file_totals(print_dir, type_files)
		print()

def print_f10_directory_totals():
	bird_parakeet = ("Feeder10_bird_parakeet", ["bird", "parakeet"])
	bird_empty = ("Feeder10_bird_empty", ["bird", "empty"])
	bird_parakeet_empty = ("Feeder10_bird_parakeet_empty", ["bird", "parakeet", "empty"])
	print_directory_totals("Feeder10_sorted/", [bird_parakeet, bird_empty, bird_parakeet_empty])

def print_f11_directory_totals():
	bird_parakeet = ("Feeder11_bird_parakeet", ["bird", "parakeet"])
	bird_empty = ("Feeder11_bird_empty", ["bird", "empty"])
	bird_parakeet_empty = ("Feeder11_bird_parakeet_empty", ["bird", "parakeet", "empty"])
	print_directory_totals("Feeder11_sorted/", [bird_parakeet, bird_empty, bird_parakeet_empty])

def main():
	#dates = ["2020-01-29 to 2020-02-20 DONE HH - filtered HH 06-04-20/", 
	#"2020-02-20 to 2020-03-04 DONE HH - filtered HH 07-04-20/",
	#"2020-03-04 to 2020-03-19 DONE KT - filtered RW 03-04-20/"]
	#dest_path = "Feeder10_sorted/Feeder10_bird_parakeet_empty/"
	#dates = ["2020-01-28 to 2020-02-20 DONE HH - filtered HH 07-04-20",
	#"2020-02-20 to 2020-03-04 DONE HH - filtered HH 07-04-20",
	#"2020-03-04 to 2020-03-18 DONE KT - filtered RW 06-04-20"]
	#dest_path = "Feeder11_sorted/Feeder11_bird_parakeet/"
	#sorts = [("NPP/Birds", "bird"), ("", "parakeet")]
	#sort_training_testing(dates, dest_path, "F011 [SBH]/", sorts)
	#sort_testing_validation("Feeder11_sorted/Feeder11_bird_parakeet/training/", ["bird", "parakeet"])
	
	
main()