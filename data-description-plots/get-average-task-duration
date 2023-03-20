import csv

csv_file = "data_set_CPU.csv"
txt_file = "number_of_machines.txt"


def main():
    with open(csv_file, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter='\t')
        header = True
        machine_list = []
        for row in datareader:
            if header:
                print(row[2])
                header = False
            elif len(row) > 3 and row[3] not in machine_list:
                machine_list.append(row[3])
        f = open(txt_file, "x")
        f.write(str(len(machine_list)))
        f.write(";")
        f.write(';'.join(machine_list))
        f.close()


if __name__ == "__main__":
    main()
